# Đề xuất Nghiên cứu & Phát triển Kiến trúc MoE cho ASR

## 1. Tổng quan Kiến trúc Hiện tại

### 1.1 CA-SAMoE (Competitive-Attractive Shared-Adapter Mixture of Experts)

Kiến trúc hiện tại của bạn là **CA-SAMoE** - một kiến trúc Mixture of Experts nâng cao cho ASR với các đặc điểm chính:

#### Thành phần cốt lõi:

1. **Encoder Backbone**:
   - Transformer hoặc Conformer (6 lớp mặc định)
   - Model dimension: 256
   - Attention heads: 4
   - Relative positional encoding

2. **SharedAdapterMoE FFN**:
   - **Trunk (shared)**: Mỗi expert có Linear(model_dim → hidden_dim=1024) + GELU + Dropout
   - **Adapter down**: Mỗi expert có Linear(hidden_dim → adapter_hidden_dim=256)
   - **Adapter up**: Mỗi expert có Linear(adapter_hidden_dim → model_dim)
   - **Router**: utterance-level pooling → Linear(model_dim → num_experts)
   - **Output**: share_down + adapter_down (skip connection với adapter)

3. **Competition-Aware Routing**:
   - Per-expert CTC loss evaluation (compute_expert_scores)
   - Competition targets: q = scores / sum(scores), fitness = sum(q)
   - KL divergence loss giữa actual gates và competition targets
   - Load balance regularizer (MSE với uniform distribution)
   - Temperature annealing scheduling

4. **Expert Evolution Mechanism**:
   - Selection parents: fitness (max) + attraction (max gains over parent)
   - Merge strategy: split-linear interpolation
   - Replacement: worst/random/redundant (based on fitness+usage)
   - Diversity check: cosine similarity threshold
   - Merge mỗi N epochs sau warmup

5. **Multi-level CTC**:
   - Intermediate CTC ở giữa encoder
   - Weight: 0.3 mặc định

6. **Regularization**:
   - Stochastic depth (layer dropout): 0.1
   - SpecAugment: freq mask (27), time mask (100)
   - Label smoothing: 0.1
   - Entropy bonus: khuyến khích diverse routing

#### Baseline DME-SIM:

- Token-level MoE projector (TokenMoEProjector)
- Smear projector (token-level với average gating)
- GRU encoder (bidirectional)
- Đơn giản hơn, không có competition/evolution

---

## 2. Phân tích Chi tiết

### 2.1 Kiến trúc Shared-Adapter

**Ưu điểm**:
- **Parameter efficiency**: Adapter nhỏ (256) so với trunk (1024), tiết kiệm ~75% parameters cho mỗi expert
- **Specialization**: Adapter cho phép mỗi expert học specialized transformations trong low-rank space
- **Stability**: Shared trunk cung cấp common representation basis, adapter fine-tune

**Điểm cần xem xét**:
- Adapter down/up là non-linear (GELU) hay linear?
- Shared trunk có thể gây interference nếu các expert quá khác biệt?

### 2.2 Utterance-level vs Token-level Routing

**Utterance-level** (CA-SAMoE):
- Router pool toàn bộ utterance: (B, T, D) → (B, D) → (B, E)
- **Ưu**: ổn định, dễ train, consistent expert choice qua sequence
- **Nhược**: mất thông tin local, không tận dụng được locality

**Token-level** (DME baseline):
- Router mỗi token: (B, T, D) → (B, T, E)
- **Ư**: linh hoạt, có thể specialized theo phonetic regions
- **Nhược**: unstable, expensive compute (compute expert scores mỗi token)

**Câu hỏi nghiên cứu**: Hybrid approach? Token-level với aux losses?

### 2.3 Competition-Aware Routing

**Mechanism**:
1. Evaluate mỗi expert independently (forced_expert)
2. Compute per-sample CTC loss → per-expert scores
3. Convert to targets: q = score / sum(score) (soft assignment)
4. KL(routing_gates || q) loss

**Lý thuyết**:
- Đây là một dạng của "credit assignment" trong MoE
- Competition loss thúc đẩy routing_net học từ performance của experts
- Fitness = sum(q) là measure của expert quality

**Điểm mạnh**:
- End-to-end differentiable (through q)
- Không cần external clustering/priors
- Self-improving qua epochs

**Hạn chế**:
- Expensive: cần forward pass mỗi expert (N+1× compute)
- Có thể overfit đến validation distribution nếu compute trên train?
- Competition batch limit (args.competition_batches) để giảm cost

### 2.4 Expert Evolution

**Algorithm**:
1. **Collect statistics**: CTC loss trên validation set cho mỗi expert
2. **Select parents**: 
   - Parent A: max fitness
   - Parent B: max attraction (gains over A)
3. **Select replacement**: worst fitness (or redundant, random)
4. **Merge**: split-linear interpolation qua flattened parameters
5. **Inject noise**: Gaussian noise scale 0.01

**Merge strategy** (split_linear):
- Flatten tất cả parameters của 2 parents
- Split tại tỷ lệ split_ratio (0.5)
- Alpha interpolation: [0, split) dùng (1-α)*A + α*B; [split, end) dùng α*A + (1-α)*B

**Ý tưởng**: Kết hợp 2 parents tốt để tạo child mới, thay thế expert yếu nhất.

**Điểm thú vị**:
- Evolutionary algorithm trong parameter space
- Diversity-preserving (attraction metric)
- Online, không cần storage buffer

**Rủi ro**:
- Merge có thể làm giảm diversity nếu parents quá gần nhau
- Không có mutation strategy ngoài noise
- Chỉ thay 1 expert mỗi epoch → slow evolution

---

## 3. Ưu điểm & Đóng góp

### 3.1 Ưu điểm:

1. **End-to-end differentiable**: Competition loss + standard CTC + aux losses
2. **Self-improving**: Evolution không cần external data
3. **Parameter efficient**: Shared adapter architecture
4. **Comprehensive**: load balance, entropy, competition, evolution
5. **Flexible**: Transformer/Conformer, multiple experts, temperature scheduling
6. **Production-ready**: Cached features, profiling, EMA, gradient checkpoint

### 3.2 Đóng góp nghiên cứu:

1. **Competition-aware routing**: Novel loss function dựa trên expert performance
2. **Attraction-based parent selection**: Đo "improvement potential" thay vì random
3. **Split-linear merge**: Smooth interpolation trong parameter space
4. **Utterance-level adapter MoE**: Kết hợp adapter với utterance routing

---

## 4. Hạn chế & Thách thức

### 4.1 Computation & Memory:

1. **Competition scoring**: N+1 forward passes mỗi interval
   - Với 6 experts, batch 12, ~7× compute tại các steps
   - args.competition_batches giới hạn (default 0 = unlimited, but docs say default 0 but check code)
   - Solution: chỉ compute trên subset, hoặc less frequent

2. **Expert evolution**:
   - Validation forward pass mỗi expert (N×)
   - Expensive nếu validation set lớn
   - Solution: cached validation features, smaller subset

3. **Memory**:
   - Adapter architecture: ~4× parameters so với dense (trunk + 3× adapters)
   - Example: encoder_dim=256, ffn_hidden=1024, adapter_hidden=256, num_experts=4
     - Dense: 256×1024 + 1024×256 = 524,288 params per FFN
     - MoE: 4×[256×1024 (trunk) + 1024×256 (down) + 256×256 (up) + 256×256 (down)] = 4×(786,432) = 3,145,728
     - ~6× parameters!
   - Tuy nhiên, router chỉ learned trên utterance-level

**Observation**: Adapter architecture thực tế không giảm parameters nhiều so với dense!

### 4.2 Training Stability:

1. **Router initialization**: Linear router với Xavier uniform?
   - Nếu gate bias = 0, uniform prior?
   - Có thể cần bias initialization theo expert count?

2. **Competition loss scale**:
   - competition_weight=0.05 vs load_balance_weight=0.01
   - KL loss magnitude có ổn không?
   - Temperature annealing helps?

3. **Expert collapse**:
   - Diversity threshold check (expert_diversity_threshold=0.0 mặc định)
   - Cosine similarity trên flattened parameters
   - Khi nào trigger? Skip merge nếu similarity > threshold

4. **Evolution scheduling**:
   - expert_evolve_start_epoch=3 (không evolve early)
   - Why 3? Có thể experts chưa stabilized?
   - Warmup epochs cần thiết?

### 4.3 Evaluation & Metrics:

1. **Metric đầy đủ**:
   - CER, WER (corpus vs mean)
   - Loss components: base, load_balance, competition, entropy
   - Expert fitness, avg_gates, routing_entropy
   - Domain-level loss

2. **Missing**:
   - Expert specialization: which expert handles which phonemes/words?
   - Computational efficiency: FLOPs, latency (không có trong code)
   - Expert quality distribution (variance của fitness)

### 4.4 Generalization:

1. **Language**: Chỉ test trên Telugu, Hinglish (IndicTTS, MUCS)
   - Không có LibriSpeech (train_libri.py có nhưng chưa chạy?)
   - Cần test trên diverse languages, scripts

2. **Domain**: Chỉ ASR task
   - Có thể áp dụng cho MT, NLU?
   - Encoder-decoder architecture cần thay đổi routing?

3. **Dataset size**: Small-medium (thousands hours?)
   - Scale to larger? LibriSpeech 1000h, GigaSpeech 10kh?
   - More experts? (8, 16, 32?)

---

## 5. Đề xuất Hướng Nghiên cứu

### 5.1 Ngắn hạn (1-6 months): Tối ưu & Phân tích

#### 5.1.1 Ablation Studies (Urgent)

**Mục tiêu**: Hiểu contribution của từng component

**Các ablation cần chạy**:

1. **Routing variant**:
   - [ ] Token-level routing (thay utterance-level)
   - [ ] No temperature annealing
   - [ ] Different pooling strategies: mean, max, first token

2. **Loss components**:
   - [ ] Remove competition loss
   - [ ] Remove load balance (so sánh với uniform)
   - [ ] Remove entropy bonus
   - [ ] Vary weights: competition_weight [0.01, 0.05, 0.1, 0.2]
   - [ ] Vary load_balance_weight [0.001, 0.01, 0.1]

3. **Architecture**:
   - [ ] Dense baseline (ffn_type=dense)
   - [ ] SharedAdapter vs Full experts (no sharing)
   - [ ] Adapter hidden dim: [64, 128, 256, 512]
   - [ ] Trunk only (no adapters) → pure MoE

4. **Evolution**:
   - [ ] No evolution
   - [ ] Different merge ratios: split_ratio [0.3, 0.5, 0.7]
   - [ ] Different strategies: worst vs random vs redundant
   - [ ] Evolution frequency: mỗi 1, 3, 5 epochs
   - [ ] Evolution start epoch: 1, 3, 5, 10

5. **Multi-level CTC**:
   - [ ] No intermediate CTC
   - [ ] Different intermediate layers: 1/4, 1/2, 3/4
   - [ ] Different weights: [0.1, 0.3, 0.5]

**Output**: Bảng so sánh CER/WER, expert usage pattern, convergence speed

#### 5.1.2 Router Analysis

**Questions**:
- Router học được gì từ pooled utterance?
- Expert specialization theo phonetic/linguistic features?
- Routing pattern ổn định qua time?

**Methods**:
1. **Expert utilization tracking**:
   ```python
   # Track per-domain, per-utterance length, speaker (if available)
   routing_stats = {
       "domain": defaultdict(list),
       "length_bins": defaultdict(list),  # <1s, 1-2s, 2-5s, >5s
       "avg_gates_per_expert": torch.zeros(num_experts)
   }
   ```

2. **Clustering analysis**:
   - Cluster utterances by routing pattern (cosine similarity của gates)
   - PCA/t-SNE trên gates distribution
   - Phoneme distribution per expert (if alignments available)

3. **Router input probing**:
   - Linear probe trên pooled hidden state để predict expert choice
   - Attribution methods (Integrated Gradients) trên router input

**Deliverable**: Visualization, expert specialization patterns

#### 5.1.3 Competition Loss Dynamics

**Analysis**:
1. **Fitness trajectory**: Track fitness của mỗi expert qua epochs
2. **Correlation**: Fitness vs actual usage (routing gates)
3. **Convergence**: Khi nào competition targets stable?
4. **Cost-benefit**: Performance gain vs compute overhead

**Experiment**:
- Log expert_scores, competition_targets mỗi interval
- Plot heatmap: epoch × expert (fitness, usage)
- Compute correlation coefficients
- Ablate competition weight và measure speed/accuracy tradeoff

**Deliverable**: Report trên effectiveness của competition loss

#### 5.1.4 Benchmark trên LibriSpeech

**Setup**:
- train-clean-100 (100h), dev-clean, dev-other, test-clean, test-other
- So sánh với baseline: Dense, Token-MoE, Smear

**Configs**:
```bash
python train_libri.py \
  --train-manifest processed_data_librispeech/manifests/train.jsonl \
  --valid-manifest processed_data_librispeech/manifests/validation.jsonl \
  --test-manifest processed_data_librispeech/manifests/test.jsonl \
  --output-dir runs/libri_ca_samoe \
  --encoder-type conformer \
  --ffn-type shared_adapter_moe \
  --num-experts 4 \
  --competition-weight 0.05 \
  --expert-evolve-every-epochs 3
```

**Metrics**:
- CER/WER trên clean/other
- Training time, memory
- Expert usage distribution
- Evolution events

**Deliverable**: LibriSpeech results, so sánh với published MoE-ASR papers

---

### 5.2 Trung hạn (6-12 months): Cải tiến Kiến trúc

#### 5.2.1 Token-level Utterance Routing (Hybrid)

**Idea**: 
- Token-level routing nhưng với global constraints (load balance)
- Hay: utterance routing nhưng token-wise weighted aggregation

**Approach**:
```python
# Current:
pooled = mask.sum(hidden, dim=1) / mask.sum(dim=1, keepdim=True)
gates = router(pooled)  # (B, E)

# Proposal 1: Token-level + attention
token_gates = router(hidden)  # (B, T, E)
gates = (token_gates * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)

# Proposal 2: Hierarchical
# First: token-level coarse routing to K experts per token
# Second: aggregate token choices → utterance decision
```

**Advantages**:
- Capture local variability
- Still efficient (no N+1 forward)

**Challenges**:
- Stability: token-level noisy
- Need stronger regularization

#### 5.2.2 Adaptive Expert Count

**Problem**: Fixed num_experts có thể inefficient
- Một số expert chưa được dùng
- Một số expert overloaded

**Solution**:
1. **Dynamic activation**: Router output top-k (k=2) thay vụ softmax
2. **Expert dropout**: Randomly disable experts mỗi batch (training)
3. **Progressive addition**: Start với 2 experts, thêm dần
4. **Sparse MoE**: Only activate subset (Đã có: softmax → implicitly sparse)

**Research questions**:
- Optimal k for this architecture?
- When to add experts?
- How to measure "expert needed"?

#### 5.2.3 Router Architecture Improvement

**Current**: Single Linear layer trên pooled hidden

**Alternatives**:
1. **Two-layer router**: Linear → ReLU → Linear
2. **Task-specific routing**: Auxiliary tasks (phoneme classification) help router?
3. **History-aware routing**: LSTM/Transformer trên previous routing decisions
4. **Noise injection**: Router noise for exploration (like E-MoE)

**Experiment**:
- Compare router architectures
- Gumbel-Softmax cho discrete routing?
- Learnable temperature

#### 5.2.4 Expert Architecture Variants

**Beyond shared adapter**:

1. **Mixture of Mixtures**:
   - Hierarchical MoE: coarse experts → fine experts
   - Different expert types trong cùng layer (CNN, LSTM, linear)

2. **Conditional computation**:
   - Early exit: nếu confidence high, skip remaining layers
   - Skip connections: some tokens skip certain experts?

3. **Weight generation**:
   - Hypernetworks generate expert weights từ input
   - LoRA-style: low-rank per expert

4. **Specialization by frequency**:
   - High-frequency phonemes → dedicated experts?
   - Learnable grouping

#### 5.2.5 Better Expert Evolution

**Current limitations**:
- Only merge 2 → 1 mỗi epoch → slow
- No splitting (only merging)
- Random choice của replacement

**Improvements**:

1. **Clonal selection**:
   - Keep top-M experts, replace bottom-N
   - Multiple children per generation

2. **Crossover + Mutation**:
   - Crossover: split-linear (current)
   - Mutation: add Gaussian noise to child
   - Orthogonal initialization cho new expert slots?

3. **Speciation**:
   - Group experts bởi similarity
   - Protect diverse niches
   - Fitness sharing

4. **Evolution strategies** (CMA-ES, PEPGG):
   - Learn merge ratio (α) từ performance
   - Adaptive mutation strength

5. **Memory replay**:
   - Store best-performing expert states
   - Reintroduce nếu current pool degenerate

---

### 5.3 Dài hạn (1-2 years): Lý thuyết & Scaling

#### 5.3.1 Theoretical Analysis

**Open questions**:

1. **Routing optimality**:
   - Under what conditions does competition loss converge?
   - Relationship với load balancing in integer programming
   - Credit assignment problem: prove competition targets correct?

2. **Expert evolution convergence**:
   - Does merging guarantee improvement?
   - Merge bias: interpolation có thể converge to mediocre?
   - Diversity vs quality trade-off (Pareto front)

3. **Generalization bounds**:
   - MoE với dynamic experts có bound trên VC dimension?
   - Overfitting risk với high-capacity experts?

**Approach**:
- Collaborate với ML theorists
- Simplify setting: linear experts, synthetic data
- Prove convergence under assumptions

#### 5.3.2 Scaling to Large Models

**Current**: encoder_dim=256, ffn_hidden=1024, num_experts=4-6

**Scaling directions**:

1. **Model size**:
   - encoder_dim: 512, 1024, 2048
   - ffn_hidden: 2048, 4096, 8192
   - Layers: 12, 24, 48

2. **Expert count**:
   - 8, 16, 32, 64, 128
   - Challenge: router capacity, compute cost (N+1 forward)
   - Solution: token-level routing (no extra forward), hierarchical routing

3. **Data scale**:
   - GigaSpeech (10k hours), MLS (50k hours)
   - Multi-lingual: 1000+ languages
   - Domain: medical, legal, conversational

4. **Compute efficiency**:
   - Expert parallelism (DataParallel hiện tại chỉ replicate toàn bộ)
   - Load balancing across GPUs: dispatch tokens to device với expert
   - Mixture of Experts inference optimization (pruning, distillation)

**Challenges**:
- Memory: 128 experts × large hidden = huge
- Router design: need more capacity
- Evolution cost: O(N) evaluation mỗi evolution step infeasible
- Sampling-based evolution? Bandit algorithms?

#### 5.3.3 Beyond ASR: Multi-task & Multi-modal

**Transfer to other architectures**:

1. **Encoder-Decoder** (Speech translation, ASR+MT):
   - Router trong decoder cross-attention FFN?
   - Different routing per layer, per task?

2. **Multi-task learning**:
   - Shared encoder, task-specific experts
   - Task routing: utterance có task ID?
   - Task competition: which expert good for which task?

3. **Multi-modal** (audio-visual ASR):
   - Different modalities → different expert pools
   - Cross-modal routing

4. **Other domains**:
   - Machine translation
   - Language modeling (GPT-style)
   - Vision (ViT-MoE)

**Experiment**: 
- Convert architecture to seq2seq
- Test trên CoVoL (speech translation), How2 (multimodal)

#### 5.3.4 Novel MoE Paradigms

**Beyond current design**:

1. **Differentiable expert creation**:
   - Instead of merge/split, learn expert parameters end-to-end
   - Soft experts: weighted combinations của prototypes

2. **Reinforcement Learning**:
   - Router như policy network
   - Reward: CTC loss improvement
   - Policy gradient: REINFORCE, PPO

3. **Meta-learning**:
   - Learn to route based on few-shot examples
   - Task-adaptive MoE

4. **Neuro-symbolic MoE**:
   - Some experts symbolic (rule-based)
   - Some experts neural
   - Router học to combine

5. **Quantum-inspired**:
   - Superposition of expert states
   - Entanglement between experts?

---

## 6. Specific Research Questions

### 6.1 Immediate (Can answer in 3-6 months):

1. **What is the marginal gain of competition loss vs load balance?**
   - Ablate each component systematically
   - Measure statistical significance

2. **How does adapter dimension affect performance?**
   - Sweep: adapter_hidden_dim ∈ [32, 64, 128, 256, 512, 1024]
   - Trade-off: parameters vs accuracy

3. **What is the optimal number of experts for LibriSpeech?**
   - Sweep: [1, 2, 4, 6, 8, 12, 16]
   - Law of diminishing returns?

4. **Does expert evolution actually improve diversity?**
   - Track cosine similarity matrix pre/post merge
   - Measure entropy of routing distribution

5. **How transferable are evolved experts across domains?**
   - Train on Indic, test on MUCS
   - Freeze experts, only train router?

### 6.2 Mid-term (6-12 months):

6. **Can token-level routing improve utterance-level?**
   - Implement hybrid approach
   - Compare stability, accuracy, compute

7. **What router architecture maximizes expert specialization?**
   - Deeper router
   - Attention-based router
   - History-aware

8. **How to scale to 16+ experts without N+1 compute?**
   - Token-level routing (no extra forward)
   - Hash-based routing (without learnable router)
   - Sparse gating (top-k)

9. **Can we predict which utterances benefit from MoE?**
   - Analyze routing pattern vs difficulty (CTC loss)
   - Build confidence estimator

10. **Is competition loss necessary with enough data?**
    - Train trên LibriSpeech 1000h vs 100h
    - Does competition help more in low-data regime?

### 6.3 Long-term (1-2 years):

11. **Theoretical: Under what conditions does expert evolution converge to optimal?**
    - Prove or disprove monotonic improvement
    - Identify failure modes

12. **Can we automatically determine optimal expert count?**
    - Bayesian optimization over num_experts
    - Validation perplexity surrogate

13. **How does MoE ASR generalize to unseen languages?**
    - Zero-shot transfer: train on 10 languages, test on 11th
    - Compare với multilingual dense model

14. **Can we compress evolved MoE models for deployment?**
    - Distill multiple experts → single dense
    - Prune redundant experts
    - Quantization impact

15. **Does competition-aware routing transfer to other losses?**
    - Instead of CTC, use sequence-to-sequence (AED)
    - Test trên speech translation

---

## 7. Practical Experiment Plan

### Phase 1: Baseline Establishment (2 weeks)

**Tasks**:
1. Reproduce current CA-SAMoE trên Indic/Telugu
2. Document hyperparameters, seeds, hardware
3. Run DME-SIM baseline với same data
4. Benchmark: time per epoch, memory peak, GPU utilization

**Deliverable**: Reproducibility report, baseline numbers

### Phase 2: Ablation Suite (4 weeks)

**Config template**:
```yaml
ablation:
  - name: "no_competition"
    args: {"competition_weight": 0.0}

  - name: "no_load_balance"
    args: {"load_balance_weight": 0.0}

  - name: "no_evolution"
    args: {"expert_evolve_every_epochs": 0}

  - name: "dense_baseline"
    args: {"ffn_type": "dense"}

  - name: "full_experts"
    args: {"ffn_hidden_dim": 256}  # No shared trunk?

  - name: "competition_weight_sweep"
    args: {"competition_weight": [0.01, 0.05, 0.1, 0.2]}

  - name: "adapter_dim_sweep"
    args: {"adapter_hidden_dim": [32, 64, 128, 256]}
```

**Execution**:
- 3 random seeds mỗi config
- Train 10 epochs (hoặc early stop)
- Log tất cả metrics mỗi epoch

**Analysis**:
- CER/WER với error bars
- Training curves (loss, competition loss, routing entropy)
- Statistical significance (t-test)

**Deliverable**: Ablation paper section, heatmaps

### Phase 3: Scaling Studies (4 weeks)

**Experiments**:

1. **Experts scaling**:
   - num_experts: [1, 2, 4, 6, 8, 12] trên LibriSpeech 100h
   - Measure: accuracy, training time, memory

2. **Model scaling**:
   - encoder_dim: [128, 256, 512]
   - ffn_hidden: [512, 1024, 2048]
   - Interactions: num_experts × model_size

3. **Data scaling**:
   - Train subsets: 10h, 25h, 50h, 100h
   - Does MoE advantage increase/decrease với data?

**Deliverable**: Scaling laws, recommendations

### Phase 4: Analysis & Visualization (2 weeks)

**Analysis tasks**:

1. **Routing pattern analysis**:
   - Cluster utterances by routing
   - Phoneme distribution per expert (nếu forced alignment)
   - Length vs expert choice

2. **Evolution tracking**:
   - Fitness trajectory plots
   - Parent-child similarity (cosine)
   - Diversity metrics over time

3. **Error analysis**:
   - Which samples fail? Routing pattern?
   - Expert overload: some experts >90%?
   - Case studies: good routing vs bad routing

**Deliverable**: Visualizations, insights report

### Phase 5: Novel Variants (8 weeks)

**Implement và test**:

1. **Token-level utterance routing** (5.2.1)
2. **Adaptive expert activation** (5.2.2) - top-k routing
3. **Improved router** (5.2.3) - 2-layer
4. **Better evolution** (5.2.5) - clonal selection

**Evaluation**: So sánh với ablation best config

**Deliverable**: Prototype implementations, results

---

## 8. Publication Strategy

### Target Venues:

1. **ASR-specific**:
   - ICASSP, Interspeech (workshop/long paper)
   - IEEE/ACM Transactions on Audio, Speech, and Language Processing

2. **ML/NLP**:
   - NeurIPS (ML for speech workshop)
   - ICLR (MoE workshop)
   - ICML

3. **Systems**:
   - MLSys (if focus on scaling/efficiency)

### Paper Ideas:

**Paper 1 (Short, <6 months)**: 
"CA-SAMoE: Competitive-Attractive Shared-Adapter Mixture of Experts for Efficient ASR"
- Present architecture, ablation trên Indic/LibriSpeech
- Emphasize parameter efficiency, competition loss

**Paper 2 (Medium, 12 months)**:
"Expert Evolution in Mixture of Experts for Speech Recognition"
- Focus on evolutionary algorithm, diversity maintenance
- Long-term training dynamics

**Paper 3 (Long, 18-24 months)**:
"Scaling Mixture of Experts to 128 Experts for Low-Resource ASR"
- Scaling laws, multi-lingual, parameter efficiency
- Theory + large-scale experiments

---

## 9. Open Questions (Discussion)

### 9.1 Architectural:

1. **Utterance vs token routing**: Which is better for ASR?
   - Token: more flexible but unstable
   - Utterance: stable but coarse
   - Could hierarchical: utterance router → token-wise weighted

2. **Shared trunk necessary?**
   - Current: trunk shared → adapter specialized
   - Alternative: full experts (no sharing) với regularization?
   - Measure: parameter count vs performance

3. **Competition loss variant?**
   - Instead of CTC-based scores, use reinforcement learning?
   - Use auxiliary losses (phone classification) để score experts?

4. **Evolution frequency?**
   - Current: every N epochs
   - Adaptive: when diversity drops below threshold?
   - Online: evolutionary algorithm mỗi batch (expensive)

### 9.2 Training:

1. **Curriculum**: Start dense, gradually introduce MoE?
2. **Warmup**: Current router temperature annealing. Should experts warmup too?
3. **Regularization**: Current uses standard techniques. MoE-specific?
4. **Optimizer**: AdamW works. Differentially update experts? (faster for some)

### 9.3 Evaluation:

1. **Metrics beyond CER/WER**:
   - Expert utilization entropy
   - Specialization score (mutual information giữa expert và phoneme)
   - Computational efficiency (FLOPs, latency)

2. **When does MoE help most?**
   - Low-resource vs high-resource?
   - Diverse acoustic conditions?
   - Long utterances?

3. **Interpretability**:
   - Can we explain why expert E chosen?
   - Are experts linguistically meaningful?

### 9.4 Theoretical:

1. **Credit assignment**: Competition loss là approximation tối ưu?
2. **Convergence**: Guaranteees với non-convex objectives?
3. **Generalization**: MoE overfit hơn dense? How to regularize?

---

## 10. Immediate Next Steps

**Week 1-2**:
- [ ] Setup LibriSpeech pipeline (process_data.py, prepare manifests)
- [ ] Run CA-SAMoE baseline (10 epochs) để get numbers
- [ ] Run DME-SIM baseline same setup
- [ ] Profile: time breakdown (data, forward, backward, competition)

**Week 3-4**:
- [ ] Implement simple ablations (no competition, no evolution, dense)
- [ ] Run 3 seeds mỗi config
- [ ] Start analysis: CER curves, routing patterns

**Week 5-6**:
- [ ] Competition weight sweep [0, 0.01, 0.05, 0.1]
- [ ] Load balance weight sweep [0, 0.001, 0.01, 0.1]
- [ ] Adapter dimension sweep [32, 64, 128, 256]

**Week 7-8**:
- [ ] Experts count sweep [1, 2, 4, 6, 8] trên LibriSpeech 100h
- [ ] Token-level routing prototype
- [ ] Document findings, prepare first report

---

## 11. References (to explore)

**MoE in ASR**:
- Google's SPARSARTM (Sparse Transformer)
- Microsoft's Mixture of Shortcut Experts
- gShard, Switch Transformer

**Competition/Cooperative Multi-agent**:
- Multi-agent RL: QMIX, MADDPG
- Cooperative game theory: Shapley value, core

**Evolutionary algorithms**:
- Genetic algorithms trong NN (NeuroEvolution)
- CMA-ES, PEPGG
- Quality Diversity (MAP-Elites)

**Adapter-based methods**:
- LoRA, AdaLoRA
- Compacter (hashing)
- Parameter-efficient fine-tuning

**Router design**:
- SIFT: Switch-Irrelevance-Free Transformer
- GShard: load balancing
- V-MoE: vision MoE

---

## 12. Summary & Recommendations

### Current State Assessment:

**Strengths**:
✅ Comprehensive system: data pipeline → training → evaluation
✅ Novel components: competition loss, evolution
✅ Multiple ablations already possible (many args)
✅ Production-ready: caching, profiling, checkpointing

**Weaknesses**:
⚠️ Competition compute expensive (N+1 forward)
⚠️ Limited evaluation (only Indic/Telugu)
⚠️ No systematic ablations published yet
⚠️ Theoretical justification unclear

### Recommended Priority:

**Tier 1 (Must do)**:
1. **Ablation studies** → understand what matters
2. **LibriSpeech benchmark** → standard comparison
3. **Router analysis** → interpretability
4. **Competition efficiency** → reduce cost

**Tier 2 (Should do)**:
5. **Scaling study** → experts count, model size
6. **Token-level routing** → potentially better
7. **Improved evolution** → faster convergence
8. **Multi-lingual** → robustness

**Tier 3 (Nice to have)**:
9. **Theory** → convergence proofs
10. **Novel variants** → different MoE paradigms
11. **Deployment** → distillation, quantization
12. **Other tasks** → MT, SLT

### Estimated Resources:

**Personnel**:
- 1 researcher (you): full-time 6-12 months
- 1 engineer (part-time): infrastructure, scaling
- 1 theorist (collaborator): optional, for theory

**Compute**:
- Baseline: 1× A100 40GB (current)
- Scaling: 4× A100 for large experiments
- Datasets: LibriSpeech 100h (small), Indic/MUCS (already processed)

**Timeline**:
- Tier 1: 3 months
- Tier 2: 6 months
- Tier 3: 12 months

---

## 13. Conclusion

Kiến trúc CA-SAMoE của bạn là một **contribution substantial** trong ASR-MoE với các innovations:
1. Shared-adapter design (parameter-efficient)
2. Competition-aware routing (self-improving)
3. Evolutionary expert management

Tuy nhiên, cần **systematic evaluation** để chứng minh advantage và **addressing limitations** để scale.

**Next immediate step**: Reproduce baseline, run ablations, publish findings.

---

**Author**: Analysis generated by Claude Code  
**Date**: 2026-04-28  
**Version**: 1.0
