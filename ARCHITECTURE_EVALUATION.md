# Đánh Giá Kiến trúc MoE Mới

**Từ**: ARCHITECTURE_REDESIGN.md  
**Ngày**: 2026-04-28  
**Người đánh giá**: Claude Code

---

## 1. Tổng quan Đánh Giá

Kiến trúc mới đề xuất là **ambitious và well-thought-out**, kết hợp nhiều advanced concepts từ recent MoE literature. Tuy nhiên, có một số **critical concerns** về computational cost và training complexity cần được addressed.

**Overall Rating**: 7.5/10
- **Innovation**: 9/10
- **Feasibility**: 6/10
- **Expected Performance Gain**: 7/10
- **Implementation Risk**: Medium-High

---

## 2. Phân tích Chi tiết theo Thành phần

### 2.1 Hierarchical Routing ⭐⭐⭐⭐ (4/5)

**Strengths**:
- ✅ **Token-level routing** đúng là cần thiết cho ASR: different phonetic segments có thể cần different experts
- ✅ **Cluster aggregation** giải quyết noise từ token-level: tokens trong cùng vùng thường có similar patterns
- ✅ **Multi-granularity** capture cả local và global information

**Weaknesses**:
- ⚠️ **Clustering cost**: O(T²) với T=time steps. Nếu T=500 (10s at 50Hz), T²=250k, expensive
- ⚠️ **Backprop through clustering** nếu dùng soft assignments: gradient flow có thể unstable
- ⚠️ **Training instability**: Thêm nhiều layers trong routing → thêm vanishing/exploding gradient

**Recommendations**:
1. **Use fixed clustering** (không learnable) để giảm complexity:
   ```python
   # Pre-compute clusters based on length/position (no gradient)
   cluster_assignments = fixed_clustering(mask)  # Stop gradient
   ```
2. **Limit clustering to training** (không cần ở inference nếu token_gates đã ổn định)
3. **Try hierarchical routing WITHOUT clustering** trước:
   - Token router → average over time → utterance gates
   - Simpler, still captures token variability

**Score**: 4/5 (good idea, cần optimize implementation)

---

### 2.2 Expert Groups ⭐⭐⭐ (3/5)

**Strengths**:
- ✅ **Encourages specialization**:phonemes → experts mapping học được rõ ràng hơn
- ✅ **Interpretability**: Có thể analyze group usage patterns
- ✅ **Load balance at group level**: Avoids one group dominating

**Weaknesses**:
- ❌ **How to define groups?** Paper thiếu chi tiết:
  - Static groups: cần phoneme classifier hoặc prior knowledge
  - Learned groups: cần supervision signal (phoneme labels?) để guide
  - Without proper grouping signal, có thể **hurt performance**
- ❌ **Group assignment overhead**: Thêm layer, thêm parameters
- ❌ **What if phoneme distribution is uniform?** Groups có thể unused

**Critical Question**: 
- **Có ground truth labels cho phoneme types trong dataset?** Nếu có, có thể supervised grouping. Nếu không, unsupervised grouping có thể **arbitrary và unhelpful**.

**Recommendations**:
1. **Start WITHOUT groups** (giữ current flat expert structure)
2. **Nếu muốn groups**, dùng **soft assignment** (mỗi expert thuộc nhiều groups) thay vì hard partition
3. **Use phone-level CTC alignments** (if available) để create pseudo-labels for grouping:
   ```python
   # From CTC forced alignment, count phoneme per expert
   phone_to_expert = defaultdict(Counter)
   for sample in dataset:
       alignment = ctc_align(sample)  # (T,) phone IDs
       expert_ids = routing_gates.argmax(dim=1)  # (T,)
       for t, (phone, exp) in enumerate(zip(alignment, expert_ids)):
           phone_to_expert[phone][exp] += 1
   # Group experts by phoneme similarity
   ```
4. **Alternative**: Skip explicit groups, use **regularization** khuyến khích specialization:
   ```python
   def specialization_entropy_loss(gates, num_experts):
       # Encourage low entropy per token (specialization)
       per_token_entropy = -(gates * torch.log(gates + 1e-8)).sum(dim=-1)
       # But high entropy globally (all experts used)
       global_entropy = -(gates.mean(dim=0) * torch.log(gates.mean(dim=0) + 1e-8)).sum()
       return per_token_entropy.mean() - 0.1 * global_entropy
   ```

**Score**: 3/5 (good concept, needs proper implementation guidance)

---

### 2.3 Load Balancing ⭐⭐⭐⭐⭐ (5/5)

**Current**: MSE(avg_gates, uniform) trên final gates

**Proposed**: Multi-level load balance

**Strengths**:
- ✅ **Comprehensive**: Monitor và balance ở token, cluster, group levels
- ✅ **Early detection**: If cluster-level imbalance detected, có thể adjust trước khi final
- ✅ **Flexible weighting**: Có thể assign different weights cho different levels

**Weaknesses**:
- ⚠️ **Over-regularization**: Quá nhiều balance losses → có thể **hurt accuracy** nếu expert specialization là good
- ⚠️ **Conflicting signals**: Token diversity vs cluster coherence có thể conflict

**Recommendations**:
1. **Start with 2-level**: token + final gates
2. **Monitor correlation** giữa các levels:
   ```python
   # If token_gates and final_gates highly correlated (>0.9),
   # cluster-level là redundant
   ```
3. **Use adaptive weights**:
   ```python
   # If cluster imbalance > threshold, increase cluster weight
   cluster_imbalance = torch.var(cluster_avg_gates)
   cluster_weight = 0.1 * (1 + 10 * cluster_imbalance)
   ```

**Score**: 5/5 (excellent improvement, well-justified)

---

### 2.4 Router Architecture ⭐⭐⭐⭐⭐ (5/5)

**Current**: Single Linear layer

**Proposed**: Hierarchical MLP or Attention

**Strengths**:
- ✅ **Deeper routers** học được complex patterns better
- ✅ **Attention router** có thể capture long-range dependencies trong sequence
- ✅ **Hierarchical router** với residual connections: easier optimization

**Weaknesses**:
- ⚠️ **Router overfitting**: Deeper routers có thể overfit nếu không đủ data
- ⚠️ **Gradient flow**: Deep router → vanishing gradients (solution: LayerNorm, residual)
- ⚠️ **Inference latency**: Attention router O(T²) với T=sequence length

**Evidence from Literature**:
- **Switch Transformer**: Sử dụng single Linear router (simple works)
- **GShard**: Two-layer MLP router
- **V-MoE**: Simple router works well nếu có good load balancing

**Critical Insight**:
Router complexity **KHÔNG** always better. Switch Transformer (Google) dùng simple Linear và đạt SOTA. **Load balancing loss** quan trọng hơn router architecture.

**Recommendations**:
1. **First try**: 2-layer MLP với LayerNorm:
   ```python
   router = nn.Sequential(
       nn.Linear(D, D),
       nn.LayerNorm(D),
       nn.GELU(),
       nn.Linear(D, E)
   )
   ```
   - Cost: ~2× parameters, minimal compute
   - Expected: small improvement

2. **Nếu MLP đủ**, try Attention chỉ nếu:
   - Long sequences (>1000 timesteps)
   - Token-level routing cần context

3. **Always compare với simple Linear**:
   - Use router complexity chỉ nếu có **statistically significant improvement**

**Score**: 5/5 (good exploration direction, cần careful experimentation)

---

### 2.5 Competition Loss ⭐⭐⭐ (3/5)

**Current**: Compute expert scores qua N+1 forward passes, KL divergence với normalized scores

**Proposed**: Keep similar, but mention "prediction-based" alternative

**Assessment**:

**Problems với Current** (chưa được address):
1. ❌ **N+1 forward passes**: Với 6 experts, 7× compute mỗi interval
2. ❌ **Memory**: Lưu tất cả expert outputs simultaneously → O(N) memory
3. ❌ **Validation cost**: Competition_on_valid=True → double validation time

**Prediction-based approach** (mentioned nhưng không detailed):
- ✅ **Good idea**: Train predictor network dự đoán expert performance
- ❌ **Chicken-egg problem**: Predictor cần expert outputs để train, nhưng goal là tránh expert forwards
- **Solution**: Train predictor offline từ early epochs, sau đó freeze

**Better alternatives** (không mentioned):
1. **Random subset scoring**: Chỉ compute competition trên random subset của batch (đã có `competition_batches`)
2. **Sampled experts**: Mỗi step, chỉ evaluate M random experts (M << N)
3. **Historical moving average**: Maintain exponential moving average của expert scores, update slowly

**Recommendations**:
1. **Don't change competition mechanism** trong redesign (đã tốt conceptually)
2. **Focus on efficiency**:
   ```python
   # Already implemented: competition_batches
   # Add: competition_sampling_ratio (e.g., 0.25 = 25% of batch)
   # Add: competition_expert_sample (e.g., 3 random experts thay vì all)
   ```
3. **Logging**: Track correlation giữa predicted scores (từ predictor) và actual scores

**Score**: 3/5 (competition concept tốt, nhưng redesign không解决 computational cost)

---

### 2.6 Expert Evolution ⭐⭐⭐⭐⭐ (5/5)

**Current**: 
- 1 expert replaced mỗi epoch
- Select parents: max fitness + max attraction
- Merge: split-linear interpolation

**Proposed**:
- ✅ **Clonal selection**: Replace N experts (N>1) → faster evolution
- ✅ **Speciation**: Fitness sharing để protect diversity
- ✅ **Adaptive frequency**: Trigger khi diversity drops

**Strengths**:
- ✅ **Multiple offspring**: Faster exploration of expert space
- ✅ **Speciation**: Prevents premature convergence (Niche protection)
- ✅ **Adaptive**: Không waste compute nếu diversity đã good

**Evidence**:
- **Evolutionary algorithms** trong ML (NeuroEvolution) shows:
  - Population size matters: More simultaneous evolution → better
  - Speciation (NEAT) maintains diversity
  - Adaptive mutation rates theo diversity

**Potential Issues**:
- ⚠️ **Multiple merges**在同一 epoch: Could destabilize training
- ⚠️ **Speciation threshold**: How to set? May need tuning
- ⚠️ **Child initialization**: Noise scale (0.01) có thể quá nhỏ

**Recommendations**:
1. **Start with `num_to_replace=2`** (not 1), giữ other nguyên
2. **Add speciation** sau khi thấy diversity collapse:
   ```python
   if mean_cosine_sim > 0.85:
       apply_speciation_sharing(fitness, similarity_matrix)
   ```
3. **Track diversity metrics** per generation:
   ```python
   metrics = {
       'mean_cosine_sim': ...,
       'num_species': ...,
       'fitness_std': ...,
   }
   ```
4. **Consider splitting** (not just merging):
   - Split worst expert thành 2 variants
   - Add small noise to each
   - Increases expert count? Or replace?

**Score**: 5/5 (excellent evolutionary improvements)

---

### 2.7 Shared-Adapter Enhancement ⭐⭐⭐ (3/5)

**Current**: Shared trunk + individual adapters

**Proposed**: Grouped adapters

```python
class GroupedSharedAdapterMoE(nn.Module):
    # Groups share trunk, experts have adapters
```

**Analysis**:
- ✅ **Intuitive**: Group-related experts share computation
- ❌ **Complexity**: Thêm group assignment layer, potential bottleneck
- ❌ **Diminishing returns**: Current shared trunk already parameter-efficient
- ❌ **What's the gain?** Paper không estimate parameter savings

**Calculation**:
```
Current (4 experts, D=256, H=1024, A=256):
- Trunk: 4 × (256×1024 + 1024) = 1,048,576
- Adapters: 4 × (1024×256 + 256×256 + 256×256) = 4 × 655,360 = 2,621,440
- Total: ~3.67M params per MoE layer

Proposed (G=2 groups):
- Group trunks: 2 × (256×1024 + 1024) = 524,288
- Adapters: same 2.62M
- Router: (256×2) + (4×2) = ~512
- Total: ~3.14M params (slight improvement)
```

**Verdict**: Slight parameter saving (~14%), nhưng thêm complexity. **Not worth it** unless groups clearly defined.

**Better approach**: **Progressive sharing**:
- Early layers: more sharing (trunk shared across all experts)
- Later layers: less sharing (each expert more specialized)
- Learned sharing ratio

**Score**: 3/5 (interesting nhưng marginal benefit)

---

## 3. Overall Architecture Coherence

### Strengths:

1. ✅ **Systematic design**: Mỗi component có clear purpose
2. ✅ **Modular**: Các components có thể được enable/disable độc lập
3. ✅ **Interpretable**: Multi-level routing cho insights vào decision process
4. ✅ **Scalable**: Có thể scale số experts, clusters, groups
5. ✅ **Based on solid principles**: Hierarchical processing (như vision transformers, NLP hierarchies)

### Weaknesses:

1. ❌ **Complexity explosion**: 
   ```
   Current: 1 router per layer
   Proposed: 5 routers + clustering + grouping per layer
   ```
   ~5× parameters cho routing logic, ~2× compute

2. ❌ **Training difficulty**:
   - More components → more failure modes
   - Gradient signals từ multi-level losses có thể conflict
   - Hyperparameter tuning nightmare

3. ❌ ** diminishing returns**:
   - Going from utterance→token: likely large gain (proven in literature)
   - Adding clusters: moderate gain
   - Adding groups: uncertain gain
   - **Law of diminishing returns**: Each additional layer of complexity yields smaller improvements

4. ❌ **Inference latency**:
   - Token-level routing: E×T expensive nếu E lớn
   - Clustering: O(T²) với attention-based methods
   - Production deployment **very challenging**

---

## 4. Comparison với Current CA-SAMoE

| Aspect | Current | Proposed | Verdict |
|--------|---------|----------|---------|
| **Routing granularity** | Utterance | Token + Cluster + Utterance | 🟡 Better but costly |
| **Router capacity** | Linear | Hierarchical MLP/Attention | 🟢 Clear upgrade |
| **Load balancing** | Single-level | Multi-level | 🟢 Improvement |
| **Expert specialization** | Emergent | Encouraged via groups | 🟡 Needs validation |
| **Evolution speed** | 1 expert/epoch | N experts + speciation | 🟢 Major upgrade |
| **Total parameters** | Baseline | +20-40% | 🟡 Acceptable |
| **Training stability** | Good | Uncertain | 🔴 Risk |
| **Inference speed** | Fast | Slower (token-level) | 🔴 Concern |
| **Implementation effort** | Done | 4-6 weeks | 🟡 Medium |
| **Expected accuracy gain** | - | +5-15% CER? | 🟢 Promising |
| **Production readiness** | Yes | Questionable | 🔴 Needs optimization |

---

## 5. Critical Questions Cần Trả Lời

### Q1: **Is token-level routing worth the cost?**

**Evidence**:
- **Pro**: Token-level routing được sử dụng trong nhiều SOTA MoE models (GShard, Switch Transformer)
- **Con**: Token-level routing **expensive** O(T×E), cần optimization (e.g., sequence length > 500 mới thấy benefit)
- **Your case**: ASR sequences ~500-1000 timesteps → token-level **might be worth it**

**Recommendation**:
- **Implement token-level WITHOUT clustering first**
- Benchmark: utterance vs token (same number of experts)
- Nếu gain > 5% CER với < 20% slowdown → worth it
- Nếu gain < 2% → stick với utterance

### Q2: **How to define expert groups without phoneme labels?**

**Problem**: Proposal assumes phonetic grouping, nhưng dataset (Indic/MUCS) có phone alignments?

**Check**:
```bash
# Inspect dataset
cat processed_data_indic/manifests/train.jsonl | head -1 | jq
# Look for: "phonemes", "phones", "alignment" fields
```

**If NO phoneme labels**:
1. **Skip explicit groups**
2. **Use unsupervised grouping**:
   ```python
   # Cluster expert parameters sau training
   from sklearn.cluster import KMeans
   expert_params = [get_expert_params(i) for i in range(E)]
   kmeans = KMeans(n_clusters=G).fit(expert_params)
   groups = kmeans.labels_
   ```
3. **Analyze groups** sau khi trained: which phonemes each group handles?
4. **Optional**: Re-train với group regularization

### Q3: **Can we train hierarchical router từ scratch?**

**Risk**: Deep routers có thể không converge với default initialization

**Solution**:
1. **Router warmup** (freeze experts, train router only, first 2-3 epochs)
2. **Pre-train router** trên small dataset, sau đó fine-tune jointly
3. **Use Gumbel-Softmax** cho differentiable discrete decisions trong clustering

### Q4: **How to tune 2× hyperparameters?**

**Problem**: Current architecture ~10 HPs, proposed adds ~10 more = 20 HPs

**Solution**:
1. **Sequential tuning**:
   - Step 1: Fix all proposed components except router type → tune router
   - Step 2: Tune clustering method
   - Step 3: Tune group count (if using groups)
   - Step 4: Tune loss weights (load balance, competition, entropy, group)

2. **Use defaults** khi có thể:
   ```python
   DEFAULT_CLUSTER_METHOD = 'similarity'
   DEFAULT_NUM_CLUSTERS = 8
   DEFAULT_NUM_GROUPS = 4
   DEFAULT_ROUTER_TYPE = 'hierarchical_mlp'
   ```

3. **Ablate systematically**:
   - Baseline: current CA-SAMoE
   - + Token routing only
   - + Token + Clustering
   - + Token + Clustering + Groups
   - + All + Advanced router
   - **Measure incremental gain** at each step

---

## 6. Implementation Priority

### Tier 1 (Must implement - Highest value, lowest risk):

1. **Token-level routing** (replace utterance-level)
   - Value: ⭐⭐⭐⭐⭐ (core improvement)
   - Risk: ⭐⭐ (well-studied, should work)
   - Effort: 2-3 days
   - **START HERE**

2. **Hierarchical router** (2-layer MLP)
   - Value: ⭐⭐⭐ (moderate improvement)
   - Risk: ⭐ (very safe)
   - Effort: 1 day
   - **DO THIS with token routing**

3. **Multi-level load balancing**
   - Value: ⭐⭐⭐⭐ (comprehensive monitoring)
   - Risk: ⭐ (just add losses)
   - Effort: 1 day

### Tier 2 (Should implement - Medium value, medium risk):

4. **Improved evolution** (clonal selection, N>1)
   - Value: ⭐⭐⭐⭐ (faster convergence)
   - Risk: ⭐⭐ (could cause instability)
   - Effort: 3-4 days

5. **Specialization regularization** (instead of groups)
   - Value: ⭐⭐⭐ (encourages diversity)
   - Risk: ⭐ (just another loss term)
   - Effort: 1 day

### Tier 3 (Nice to have - Uncertain value):

6. **Expert groups** (if phoneme labels available)
   - Value: ⭐⭐⭐ (interpretable)
   - Risk: ⭐⭐⭐ (may not help)
   - Effort: 1 week (including analysis)

7. **Attention-based router**
   - Value: ⭐⭐ (maybe for long sequences)
   - Risk: ⭐⭐⭐ (slow, may overfit)
   - Effort: 2-3 days

8. **Curriculum routing**
   - Value: ⭐⭐ (theoretical)
   - Risk: ⭐⭐⭐ (unproven)
   - Effort: 3-4 days

---

## 7. Recommended Implementation Plan

### Week 1-2: Token-level Foundation

**Goal**: Replace utterance-level với token-level routing

**Tasks**:
1. Modify `SharedAdapterMoEFFN.forward()`:
   - Remove pooled_hidden → router
   - Apply router to each token: `token_gates = router(hidden_states)`
   - Weighted sum: `output = torch.sum(expert_outputs * token_gates.unsqueeze(-1), dim=2)`
   
2. Update `compute_expert_scores()`:
   - Currently uses pooled representation → CTC loss
   - Token-level: compute CTC loss với token_gates? No, competition still utterance-level
   - **Keep competition utterance-based**: Pool token_gates → utterance_gates for scoring

3. Test on small subset (100 samples):
   - Check: Can it train without NaNs?
   - Monitor: Expert usage distribution
   - Baseline: Utterance-level results

**Success criteria**:
- Training loss decreases
- No gradient explosion/vanishing
- Expert usage not collapsed

---

### Week 3: Enhance Router Capacity

**Goal**: Upgrade router architecture

**Tasks**:
1. Implement `HierarchicalRouter` (2-layer MLP + LayerNorm)
2. Optional: `AttentionRouter` (only if sequences > 1000 timesteps)
3. Compare:
   - Token-level + Linear router (baseline from Week 1)
   - Token-level + Hierarchical router
   - Measure: convergence speed, final CER

**Success criteria**:
- Faster convergence OR better CER
- < 5% increase in inference time

---

### Week 4: Multi-level Load Balance

**Goal**: Add cluster-level + token-level balance

**Tasks**:
1. Implement `cluster_formation()` (simple: length-based first)
2. Compute cluster_gates từ token_gates
3. Add cluster balance loss
4. Add token balance loss (optional)
5. Tune weights: `lb_token`, `lb_cluster`, `lb_final`

**Success criteria**:
- More balanced expert usage (lower variance)
- No CER degradation (> 2% relative)

---

### Week 5: Evolution Enhancements

**Goal**: Faster expert evolution

**Tasks**:
1. Modify `evolve_experts()`:
   - Replace worst 2-3 experts instead of 1
   - Use top-3 parents, generate multiple children
2. Add speciation check:
   - Compute cosine similarity matrix
   - If mean_sim > 0.85, apply fitness sharing
3. Log diversity metrics per epoch

**Success criteria**:
- Diversity maintained longer (lower expert similarity)
- Faster improvement in validation CER

---

### Week 6: Integration & Ablation

**Goal**: Comprehensive evaluation

**Tasks**:
1. Combine all Tier 1 improvements
2. Run full ablation:
   ```
   Config A: Current CA-SAMoE (baseline)
   Config B: + Token routing
   Config C: + Hierarchical router
   Config D: + Multi-level LB
   Config E: + Improved evolution
   Config F: All of above
   ```
3. 3 random seeds mỗi config
4. Statistical significance test (t-test)

**Deliverable**:
- Table: CER/WER vs config
- Graph: Convergence curves
- Analysis: Which component contributes most?
- Recommendation: Final architecture

---

## 8. Cost-Benefit Analysis

### Expected Benefits (Quantitative):

| Metric | Current | Proposed (est.) | Gain |
|--------|---------|-----------------|------|
| CER (Indic) | ~15% | 13-14% | -10-20% |
| CER (LibriSpeech 100h) | ~8% | 7-7.5% | -6-12% |
| Training epochs to converge | 15 | 10-12 | -20-30% |
| Expert diversity (entropy) | 0.8 | 1.0-1.2 | +25-50% |
| Parameter count | Baseline | +20% | - |

### Expected Costs:

| Metric | Current | Proposed | Cost |
|--------|---------|----------|------|
| Training time/epoch | 1× | 1.3-1.5× | +30-50% |
| Inference latency | 1× | 1.5-2× | +50-100% |
| Memory (router) | Negligible | +10-15% | Small |
| Implementation effort | Done | 4-6 weeks | Significant |
| Tuning complexity | 10 HPs | 20 HPs | Harder |

**Net assessment**: 
- **Research**: Worth it (better understanding, potential publication)
- **Production**: May not be worth inference cost. Consider:
  - Distill hierarchical MoE → dense model
  - Use token-level only during training, distill to utterance-level

---

## 9. Alternative: Simpler Improvements

**Before committing to full hierarchical**, consider:

### 9.1 Better Utterance Router (Minimal change)

```python
class ImprovedUtteranceRouter(nn.Module):
    def __init__(self, model_dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim, num_experts)
        )
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, pooled):
        logits = self.net(pooled)
        return torch.softmax(logits / self.temperature, dim=-1)
```

**Effort**: 1 day  
**Expected gain**: +1-2% CER (from better router capacity)

---

### 9.2 Auxiliary Router Losses (No architecture change)

Add:
- **Router entropy regularization**: Encourage exploration early
- **Router contrastive loss**: Encourage different utterances to use different experts
- **Router consistency loss**: Augmented versions of same sample → similar routing

**Effort**: 2-3 days  
**Expected gain**: +1-3% CER

---

### 9.3 Progressive MoE (Curriculum)

```python
# Start with 2 experts, gradually add
if epoch < 5:
    active_experts = 2
elif epoch < 10:
    active_experts = 4
else:
    active_experts = 6
```

**Effort**: 1 day  
**Expected gain**: More stable training, similar final performance

---

## 10. Final Recommendations

### ✅ **DO**:

1. **Implement token-level routing** (Week 1-2)
   - Highest expected return
   - Well-studied in literature
   
2. **Upgrade router to 2-layer MLP** (Week 3)
   - Almost free (minimal cost)
   - Should help any routing scheme

3. **Multi-level load balancing** (Week 4)
   - Good monitoring, small benefit
   - Easy to add incrementally

4. **Improved evolution** (Week 5)
   - Speeds up expert optimization
   - Low risk if tuned conservatively

5. **Systematic ablation** (Week 6)
   - Quantify each component's contribution
   - Publishable results

### ❌ **DON'T**:

1. **Don't implement expert groups** unless you have phoneme labels
   - Unclear benefit
   - Adds complexity
   
2. **Don't use attention-based router** initially
   - O(T²) cost unacceptable for long sequences
   - Try only if MLP router insufficient

3. **Don't add clustering** early
   - Expensive, complex
   - Try token-level WITHOUT clustering first
   - Only add if token-level too noisy

4. **Don't change competition mechanism**
   - Current is conceptually sound
   - Focus on efficiency (already have `competition_batches`)
   - Don't add predictor (chicken-egg)

5. **Don't tune all HPs at once**
   - Use sequential ablation
   - Default values for new components
   - Only tune if clear improvement

---

## 11. Success Metrics

**Minimum Viable Success** (ship it):
- ✅ CER improvement > 3% relative over baseline
- ✅ Training stable (no NaNs, reasonable loss curves)
- ✅ Inference overhead < 50%
- ✅ Can train in < 2× time

**Good Success** (publishable):
- ✅ CER improvement > 8% relative
- ✅ Expert diversity ↑ 30%
- ✅ Evolution converges 30% faster
- ✅ Multi-level balance metrics show clear patterns

**Excellent Success** (top-tier):
- ✅ CER improvement > 12% on LibriSpeech 100h
- ✅ Token routing captures phonetic patterns (analysis)
- ✅ Ablation shows each component contributes
- ✅ System beats published MoE-ASR models

---

## 12. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Training instability (NaNs) | Medium | High | Gradient clipping, router warmup, start with lower LR |
| No accuracy improvement | High | Medium | Systematic ablation to identify useless components |
| Inference too slow | High | High | Top-k sparsity, expert caching, distillation |
| HP tuning nightmare | High | Medium | Sequential tuning, strong defaults, early stopping |
| Code complexity | High | Medium | Modular design, thorough testing per component |
| Overfitting | Medium | Medium | Stronger regularization (dropout, weight decay) |
| Memory OOM | Medium | High | Gradient checkpointing, smaller batch, offload |

---

## 13. Conclusion

**The proposed hierarchical MoE architecture is promising but over-engineered**.

**Recommended approach**:
1. **Start with token-level routing + hierarchical MLP router** (2-3 days)
2. **Add multi-level load balance** (1 day)
3. **Test thoroughly** before adding more
4. **Ablate systematically** to identify valuable components

**Avoid**:
- Clustering (too expensive)
- Expert groups (unclear benefit without labels)
- Attention routers (slow)
- Prediction-based competition (unnecessary)

**Expected outcome**:
- Moderate improvement (+5-10% CER)
- Better understanding of hierarchical routing
- Publishable ablation study

**Philosophy**: 
> "Make it work, then make it better, then make it faster"
> 
> **Currently**: Token-level routing is "make it better"
> **Skip**: Clustering, groups, attention until proven necessary

---

**File**: `ARCHITECTURE_EVALUATION.md`  
**Rating**: 7.5/10  
**Recommendation**: Proceed with Tier 1 components only, evaluate, iterate
