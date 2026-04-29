# Thiết kế Kiến trúc MoE Mới dựa trên Phân tích Diagram

## Tổng quan

Kiến trúc mới kết hợp **hierarchical routing** với **hybrid token-utterance** approach, **specialized expert groups**, và **improved load balancing** dựa trên insights từ diagram.

---

## 1. Kiến trúc tổng thể

```
Input Audio → Feature Extraction → Encoder Backbone
                                    ↓
                    ┌───────────────┴───────────────┐
                    │   Hierarchical MoE Layers     │
                    │  (Multi-level, Multi-group)  │
                    └───────────────┬───────────────┘
                                    ↓
                              Output Projection → CTC Loss
```

---

## 2. Hierarchical MoE Block Architecture

### 2.1 Overview

Mỗi MoE block bao gồm:

```
Hidden States (B, T, D)
        ↓
┌─────────────────────────────────────────────┐
│ 1. Token-level Router Network              │
│    - Deep MLP (2-3 layers)                 │
│    - Per-token gating: (B, T, E)           │
│    - Output: token_gates, token_scores     │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│ 2. Cluster Formation                       │
│    - Group tokens thành K clusters         │
│    - Methods:                              │
│      * Length-based bins                   │
│      * Routing pattern similarity          │
│      * Position-based (local windows)      │
│    - Output: cluster_assignments (B, T, K)│
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│ 3. Cluster-level Aggregation               │
│    - Pool token gates theo clusters        │
│    - cluster_gates = weighted_avg(token)   │
│    - Output: cluster_gates (B, K, E)       │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│ 4. Expert Group Router                     │
│    - Experts được group theo specialty    │
│    - G groups (e.g., phoneme, silence,    │
│      consonant, vowel, noise)             │
│    - Route clusters → expert groups       │
│    - Output: group_gates (B, K, G)        │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│ 5. Utterance-level Coordinator             │
│    - Pool cluster_gates → utterance vec   │
│    - Compute global expert availability  │
│    - Load balance prediction              │
│    - Output: utterance_context (B, D)     │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│ 6. Final Expert Selection                  │
│    - Combine signals:                    │
│      final_gates = f(token, cluster,      │
│                       group, utterance)   │
│    - Load balancing loss                 │
│    - Optional: top-k sparsity (k=2)      │
└─────────────────┬───────────────────────────┘
                  ↓
         Expert Execution (weighted sum)
```

---

## 3. Chi tiết Các Thành phần

### 3.1 Token-level Router

**Input**: hidden_states (B, T, D)

**Architecture**:
```python
class TokenRouter(nn.Module):
    def __init__(self, model_dim, num_experts, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or model_dim * 2
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts)
        )
        self.temperature = 1.0
        
    def forward(self, hidden_states):
        # hidden_states: (B, T, D)
        logits = self.net(hidden_states)  # (B, T, E)
        gates = torch.softmax(logits / self.temperature, dim=-1)
        return gates, logits
```

**Output**:
- `token_gates`: (B, T, E) - soft assignment mỗi token
- `token_scores`: (B, T, E) - raw logits

**Advantages**:
- Captures local phonetic patterns
- Different tokens có thể chọn experts khác nhau
- More flexible than utterance-level

---

### 3.2 Cluster Formation Module

**Mục tiêu**: Giảm computational cost của token-level routing, tạo coherence trong routing decisions.

**Input**: token_gates (B, T, E), hidden_states (B, T, D)

**Strategies**:

#### A. Length-based Clustering
```python
def length_based_clusters(token_gates, num_clusters=4):
    # Group tokens theo utterance length bins
    # Short (< 1s), Medium (1-2s), Long (2-5s), Very Long (>5s)
    B, T, E = token_gates.shape
    cluster_assignments = torch.zeros(B, T, num_clusters)
    # ... implementation
    return cluster_assignments
```

#### B. Similarity-based Clustering (Preferred)
```python
def similarity_based_clusters(token_gates, num_clusters=8):
    # Compute pairwise cosine similarity của token gates
    # K-means clustering trên token gate vectors
    # Returns: cluster_assignments (B, T, K)
    pass
```

#### C. Sliding Window Clustering
```python
def window_based_clusters(token_gates, window_size=10, stride=5):
    # Overlapping windows for local coherence
    # Returns: cluster_assignments (B, T, K) with soft assignments
    pass
```

**Output**: `cluster_assignments` (B, T, K) where K << T

---

### 3.3 Cluster-level Aggregation

**Mục tiêu**: Tổng hợp token-level decisions thành cluster-level gates.

```python
def aggregate_cluster_gates(token_gates, cluster_assignments):
    # token_gates: (B, T, E)
    # cluster_assignments: (B, T, K) - one-hot hoặc soft
    
    # Weighted average
    cluster_gates = torch.einsum('btk,bte->bke', 
                                  cluster_assignments, 
                                  token_gates)
    
    # Normalize by cluster size
    cluster_sizes = cluster_assignments.sum(dim=1, keepdim=True)  # (B, 1, K)
    cluster_gates = cluster_gates / cluster_sizes.clamp_min(1e-8)
    
    return cluster_gates  # (B, K, E)
```

---

### 3.4 Expert Group Router

**Concept**: Experts được partition thành G groups, mỗi group specializing trong một "domain".

**Grouping Strategies**:

1. **Static grouping** (pre-defined):
   - Group 0: Silence/background sounds
   - Group 1: Consonants
   - Group 2: Vowels
   - Group 3: Numeric/digits
   - Group 4: Special characters

2. **Dynamic grouping** (learned):
   - Learn group assignment matrix: (E, G)
   - Router học to distribute clusters → groups

**Implementation**:
```python
class ExpertGroupRouter(nn.Module):
    def __init__(self, num_experts, num_groups, model_dim):
        super().__init__()
        self.num_groups = num_groups
        self.group_experts = nn.Parameter(torch.randn(num_groups, num_experts))
        # Soft assignment: mỗi expert thuộc nhiều groups
        self.group_gates = nn.Linear(model_dim, num_groups)
        
    def forward(self, cluster_gates, utterance_context):
        # cluster_gates: (B, K, E)
        # utterance_context: (B, D)
        
        # Compute group assignment
        group_logits = self.group_gates(utterance_context)  # (B, G)
        group_weights = torch.softmax(group_logits, dim=-1)  # (B, G)
        
        # Distribute cluster gates to groups
        group_scores = torch.einsum('bke,eg->bkg', 
                                     cluster_gates, 
                                     self.group_experts)
        group_gates = torch.softmax(group_scores, dim=-1)
        
        return group_gates, group_weights  # (B, K, G), (B, G)
```

---

### 3.5 Utterance-level Coordinator

**Mục tiêu**: Cung cấp global context, compute load balance targets.

```python
class UtteranceCoordinator(nn.Module):
    def __init__(self, model_dim, num_experts):
        super().__init__()
        self.global_pool = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, num_experts)
        )
        self.load_balance_predictor = nn.Linear(model_dim, num_experts)
        
    def forward(self, cluster_gates, hidden_states, mask):
        # Pool over clusters and time
        pooled = (hidden_states * mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        
        # Global expert preference
        global_gates = torch.softmax(self.global_pool(pooled), dim=-1)
        
        # Load balance prediction
        load_pred = torch.sigmoid(self.load_balance_predictor(pooled))
        
        return global_gates, load_pred  # (B, E), (B, E)
```

---

### 3.6 Final Expert Selection

**Combining Multi-level Signals**:

```python
class HierarchicalMoE(nn.Module):
    def forward(self, hidden_states, mask):
        B, T, D = hidden_states.shape
        
        # 1. Token-level
        token_gates, token_logits = self.token_router(hidden_states)
        
        # 2. Cluster formation
        cluster_assignments = self.cluster_formation(token_gates, hidden_states)
        
        # 3. Cluster aggregation
        cluster_gates = self.aggregate_cluster_gates(token_gates, cluster_assignments)
        
        # 4. Expert groups
        utterance_context = (hidden_states * mask.unsqueeze(-1)).sum(dim=1)
        utterance_context = utterance_context / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        group_gates, group_weights = self.expert_group_router(cluster_gates, utterance_context)
        
        # 5. Utterance coordinator
        global_gates, load_pred = self.utterance_coordinator(cluster_gates, hidden_states, mask)
        
        # 6. Combine signals
        # Weighted combination
        final_gates = (
            0.3 * token_gates.mean(dim=1) +  # Average token gates
            0.3 * cluster_gates.mean(dim=1) + # Average cluster gates
            0.2 * group_gates.mean(dim=1) +   # Average group gates
            0.2 * global_gates                # Global gates
        )
        
        # Option: top-k sparsity
        if self.training and self.top_k > 0:
            topk_values, topk_indices = torch.topk(final_gates, k=self.top_k, dim=-1)
            final_gates = torch.zeros_like(final_gates).scatter_(-1, topk_indices, topk_values)
            final_gates = final_gates / final_gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        
        # Execute experts
        expert_outputs = [expert(hidden_states) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=2)  # (B, T, E, D)
        output = torch.sum(stacked * final_gates.unsqueeze(1).unsqueeze(-1), dim=2)
        
        aux = {
            'token_gates': token_gates,
            'cluster_assignments': cluster_assignments,
            'cluster_gates': cluster_gates,
            'group_gates': group_gates,
            'group_weights': group_weights,
            'global_gates': global_gates,
            'final_gates': final_gates,
            'load_pred': load_pred,
        }
        
        return output, final_gates, aux
```

---

## 4. Loss Functions

### 4.1 Multi-level Load Balancing

**Current**: MSE(avg_gates, uniform)

**Proposed**: Multi-granularity balance

```python
def hierarchical_load_balance_loss(aux, num_experts, weights):
    losses = []
    
    # Token-level diversity
    token_avg = aux['token_gates'].mean(dim=1).mean(dim=0)  # (E,)
    token_target = torch.ones_like(token_avg) / num_experts
    losses.append(weights['token'] * F.mse_loss(token_avg, token_target))
    
    # Cluster-level diversity
    cluster_avg = aux['cluster_gates'].mean(dim=1).mean(dim=0)  # (E,)
    losses.append(weights['cluster'] * F.mse_loss(cluster_avg, token_target))
    
    # Final gates diversity
    final_avg = aux['final_gates'].mean(dim=0)  # (E,)
    losses.append(weights['final'] * F.mse_loss(final_avg, token_target))
    
    # Group usage diversity (encourage all groups used)
    if 'group_weights' in aux:
        group_avg = aux['group_weights'].mean(dim=0)
        group_target = torch.ones_like(group_avg) / group_avg.numel()
        losses.append(weights['group'] * F.mse_loss(group_avg, group_target))
    
    return sum(losses)
```

---

### 4.2 Competition-Aware Routing (Enhanced)

**Current**: Compute expert scores with N+1 forward passes

**Proposed**: More efficient alternatives

#### Option A: Buffer-based Competition (Current)
- Keep current approach but limit to `competition_batches`
- Cache expert scores over multiple steps

#### Option B: Prediction-based Competition
```python
class CompetitionPredictor(nn.Module):
    """Predict expert performance without full forward pass"""
    def __init__(self, model_dim, num_experts):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, num_experts)
        )
        
    def forward(self, utterance_embeddings):
        # utterance_embeddings: (B, D)
        return torch.softmax(self.predictor(utterance_embeddings), dim=-1)

# Train predictor to mimic actual expert performance
```

---

### 4.3 Auxiliary Losses

1. **Cluster Consistency Loss**:
   ```python
   def cluster_consistency_loss(cluster_assignations, hidden_states):
       # Tokens trong cùng cluster nên có similar representations
       # Encourage clustering based on semantic similarity
       pass
   ```

2. **Group Specialization Loss**:
   ```python
   def group_specialization_loss(group_gates, phoneme_labels):
       # If phoneme labels available, encourage
       # specific groups handle specific phoneme types
       pass
   ```

3. **Routing Entropy Bonus** (giữ nguyên):
   ```python
   entropy = -(final_gates * torch.log(final_gates + 1e-8)).sum(dim=-1).mean()
   loss_entropy = -entropy_bonus_weight * entropy  # Negative to encourage entropy
   ```

---

## 5. Expert Architecture

### 5.1 Shared-Adapter MoE (Giữ nguyên)

**Current design** is already parameter-efficient:
- Shared trunk (large hidden)
- Individual adapters (small hidden)
- Skip connection với adapter

**Proposed enhancement**: Grouped adapters
```python
class GroupedSharedAdapterMoE(nn.Module):
    def __init__(self, model_dim, hidden_dim, adapter_dim, 
                 num_experts, num_groups):
        super().__init__()
        self.num_groups = num_groups
        
        # Group-shared trunks
        self.group_trunks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_groups)
        ])
        
        # Expert-specific adapters
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, model_dim)
            ) for _ in range(num_experts)
        ])
        
        # Router: group assignment per expert
        self.expert_to_group = nn.Parameter(torch.randn(num_experts, num_groups))
        
    def forward(self, hidden_states, routing_gates):
        # routing_gates: (B, E)
        # Determine group for each sample
        group_weights = routing_gates @ self.expert_to_group  # (B, G)
        group_idx = group_weights.argmax(dim=-1)
        
        # Use group trunk
        outputs = []
        for b in range(hidden_states.size(0)):
            trunk_out = self.group_trunks[group_idx[b]](hidden_states[b])
            # Weighted sum of adapters
            expert_out = sum(
                routing_gates[b, e] * self.adapters[e](trunk_out)
                for e in range(self.num_experts)
            )
            outputs.append(expert_out)
            
        return torch.stack(outputs, dim=0)
```

---

## 6. Expert Evolution (Improved)

### 6.1 Current Algorithm Review

**Strengths**:
- Fitness-based parent selection
- Attraction metric for diversity
- Split-linear merge

**Weaknesses**:
- Only replace 1 expert per epoch → slow
- No splitting (only merging)
- Random replacement choice

### 6.2 Proposed Enhancements

#### A. Clonal Selection with Multiple Offspring
```python
def evolve_experts_clonal_selection(fitness, usage, num_to_replace=3):
    """
    Replace worst N experts with children from top-M parents
    """
    # Select top-M parents (M > 2)
    top_m = min(5, fitness.numel() // 2)
    parent_candidates = torch.topk(fitness, top_m).indices.tolist()
    
    # Generate multiple children
    children = []
    for i in range(num_to_replace):
        # Random parent pair from top-M
        p1, p2 = random.sample(parent_candidates, 2)
        child = merge_experts_split_linear(p1, p2, alpha=random.uniform(0.3, 0.7))
        children.append(child)
    
    # Replace worst N
    worst_experts = torch.topk(fitness, num_to_replace, largest=False).indices.tolist()
    
    return worst_experts, children
```

#### B. Speciation-based Protection
```python
def calculate_speciation_fitness(fitness, similarity_matrix, species_threshold=0.8):
    """
    Apply fitness sharing: experts trong cùng species chia sẻ fitness
    """
    # Build species clusters
    species = {}
    for i in range(similarity_matrix.size(0)):
        assigned = False
        for species_id, members in species.items():
            # Check if similar to any member
            max_sim = max(similarity_matrix[i, m] for m in members)
            if max_sim > species_threshold:
                species[species_id].append(i)
                assigned = True
                break
        if not assigned:
            species[len(species)] = [i]
    
    # Apply sharing: fitness_i /= species_size
    shared_fitness = fitness.clone()
    for members in species.values():
        shared_fitness[members] /= len(members)
    
    return shared_fitness
```

#### C. Adaptive Evolution Frequency
```python
def should_evolve_adaptive(diversity_metrics, epoch, threshold=0.9):
    """
    Evolution when diversity drops below threshold
    """
    mean_sim = diversity_metrics['mean_cosine_sim']
    if mean_sim > threshold:
        return True
    return False
```

---

## 7. Router Architecture Alternatives

### 7.1 Current: Single Linear Layer

```python
router = nn.Linear(model_dim, num_experts)
```

**Problems**:
- Too simple, limited capacity
- Cannot capture complex patterns

### 7.2 Proposed: Hierarchical Router

```python
class HierarchicalRouter(nn.Module):
    def __init__(self, model_dim, num_experts, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim if i == 0 else model_dim // 2, 
                         model_dim // 2),
                nn.LayerNorm(model_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for i in range(num_levels)
        ])
        self.output = nn.Linear(model_dim // 2, num_experts)
        
    def forward(self, x):
        # x: (B, D) pooled features
        for level in self.levels:
            x = level(x)
        return torch.softmax(self.output(x), dim=-1)
```

### 7.3 Attention-based Router

```python
class AttentionRouter(nn.Module):
    def __init__(self, model_dim, num_experts, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self attention = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(num_experts, model_dim))
        self.output = nn.Linear(model_dim, 1)
        
    def forward(self, hidden_states, mask):
        # hidden_states: (B, T, D)
        # mask: (B, T)
        
        # Attend over sequence với expert queries
        queries = self.query.unsqueeze(0).expand(hidden_states.size(0), -1, -1)
        attended, _ = self.attention(
            queries, hidden_states, hidden_states,
            key_padding_mask=~mask.bool()
        )
        # (B, E, D) → (B, E)
        scores = self.output(attended).squeeze(-1)
        return torch.softmax(scores, dim=-1)
```

---

## 8. Training Strategies

### 8.1 Curriculum Routing

**Idea**: Start with simple routing, gradually introduce complexity.

```python
class CurriculumRouter(nn.Module):
    def __init__(self, model_dim, num_experts):
        super().__init__()
        self.simple_router = nn.Linear(model_dim, num_experts)
        self.complex_router = HierarchicalRouter(model_dim, num_experts)
        self.curriculum_step = 0
        self.curriculum_epochs = 3
        
    def forward(self, x, epoch):
        # Gradually switch from simple to complex
        progress = min(1.0, epoch / self.curriculum_epochs)
        if random.random() < progress:
            return self.complex_router(x)
        return torch.softmax(self.simple_router(x), dim=-1)
```

### 8.2 Router Warmup

- First N epochs: fix router, only train experts
- Then unfreeze router with lower LR
- Stabilizes training, prevents premature collapse

### 8.3 Expert Specialization Regularization

```python
def specialization_regularization(gates, targets, diversity_weight=0.1):
    """
    Encourage experts to specialize on different patterns
    """
    # Compute expert covariance
    batch_size = gates.size(0)
    expert_cov = torch.cov(gates.T)  # (E, E)
    
    # Encourage off-diagonal to be small (orthogonal specializations)
    off_diag = expert_cov - torch.diag(torch.diag(expert_cov))
    specialization_loss = (off_diag ** 2).sum() / (expert_cov.numel() - expert_cov.size(0))
    
    return diversity_weight * specialization_loss
```

---

## 9. Inference Optimizations

### 9.1 Dynamic Expert Pruning

```python
def prune_experts_during_inference(gates, threshold=0.01):
    """
    Only activate experts với gate > threshold
    """
    mask = gates > threshold
    active_experts = mask.sum(dim=-1).float().mean().item()
    
    if active_experts < gates.size(-1) * 0.5:
        # Significant pruning possible
        pruned_gates = gates.clone()
        pruned_gates[~mask] = 0
        pruned_gates = pruned_gates / pruned_gates.sum(dim=-1, keepdim=True)
        return pruned_gates, True
    
    return gates, False
```

### 9.2 Expert Caching

- Cache expert outputs for frequently occurring patterns
- Use hash-based lookup for similar inputs
- Reduces compute at inference

---

## 10. Hyperparameters Đề xuất

| Parameter | Current | Proposed | Reason |
|-----------|---------|----------|--------|
| `router_type` | linear | hierarchical | More capacity |
| `num_clusters` | N/A | 8 | Balance locality vs cost |
| `num_expert_groups` | N/A | 4 | Specialization |
| `top_k` | all | 2 (optional) | Sparsity for efficiency |
| `cluster_method` | N/A | similarity | Semantic coherence |
| `load_balance_components` | final_gates only | multi-level | Comprehensive balance |
| `router_hidden_dim` | N/A | model_dim * 2 | Capacity |
| `router_num_layers` | 1 | 2-3 | Depth |
| `competition_weight` | 0.05 | 0.02-0.05 | Lower due to multi-level |
| `entropy_bonus_weight` | 0.0 | 0.01-0.02 | Encourage exploration |

---

## 11. Expected Benefits

### 11.1 Accuracy
- ✅ Token-level routing: capture local phonetic patterns
- ✅ Cluster coherence: stable routing decisions
- ✅ Expert specialization: better utilization
- ✅ Multi-level context: richer representation

### 11.2 Efficiency
- ⚠️ Cluster formation: O(T²) for similarity, but T small after subsampling
- ⚠️ More routers: increased parameter count (~10-20%)
- ✅ Optional top-k: reduced compute at inference
- ✅ Expert caching: potential speedup

### 11.3 Training Stability
- ✅ Multi-level routing: smoother gradients
- ✅ Cluster aggregation: reduces noise from token-level
- ✅ Hierarchical structure: easier to optimize
- ⚠️ More components: more hyperparameters to tune

---

## 12. Implementation Roadmap

### Phase 1: Core Components (Week 1-2)
- [ ] Implement `TokenRouter`
- [ ] Implement `ClusterFormation` (similarity-based)
- [ ] Implement `ClusterAggregation`
- [ ] Implement `ExpertGroupRouter`
- [ ] Integrate into `SharedAdapterMoEFFN`

### Phase 2: Loss Functions (Week 3)
- [ ] Implement hierarchical load balance loss
- [ ] Add cluster consistency loss
- [ ] Add group specialization loss
- [ ] Test on small dataset

### Phase 3: Router Variants (Week 4)
- [ ] Implement `HierarchicalRouter`
- [ ] Implement `AttentionRouter`
- [ ] Implement `CurriculumRouter`
- [ ] Benchmark router capacities

### Phase 4: Evolution Enhancements (Week 5)
- [ ] Implement clonal selection
- [ ] Add speciation protection
- [ ] Implement adaptive evolution trigger
- [ ] Compare with current evolution

### Phase 5: Integration & Testing (Week 6)
- [ ] Replace current MoE block with hierarchical
- [ ] Update training script arguments
- [ ] Run ablation studies
- [ ] Benchmark against baseline

---

## 13. Ablation Studies Cần Thực Hiện

1. **Token vs Utterance routing**:
   - Pure token-level
   - Pure utterance-level (current)
   - Hierarchical hybrid (proposed)

2. **Cluster formation strategies**:
   - Length-based
   - Similarity-based
   - Window-based
   - No clustering (baseline)

3. **Expert grouping**:
   - No groups (all experts independent)
   - Static grouping
   - Learned grouping
   - Adaptive grouping

4. **Load balancing variants**:
   - Final gates only (current)
   - Multi-level (proposed)
   - With group diversity
   - Without

5. **Router architectures**:
   - Single Linear (current)
   - Hierarchical MLP
   - Attention-based
   - Curriculum

6. **Evolution algorithms**:
   - Current (1 expert/epoch)
   - Clonal selection (N experts/epoch)
   - With speciation
   - Adaptive frequency

---

## 14. Potential Issues & Solutions

### Issue 1: Computational Cost

**Problem**: Token-level routing + clustering expensive
- Token router: O(T × D × E)
- Clustering: O(T² × E) for similarity

**Solutions**:
- Downsample T (stride 2-4) before clustering
- Use approximate clustering (MiniBatchKMeans)
- Cache cluster assignments if patterns repeat

### Issue 2: Training Stability

**Problem**: Multi-level routing → complex gradients

**Solutions**:
- Gumbel-Softmax for discrete decisions in clustering
- Straight-through estimator
- Router warmup: freeze router early epochs
- Gradient clipping per component

### Issue 3: Hyperparameter Explosion

**Problem**: Many new components → many HP to tune

**Solutions**:
- Default sensible values (see Table above)
- Sequential HP search: first tune router, then clustering, then groups
- Use validation metrics to auto-tune some (e.g., cluster count)
- Start simple: disable some components initially

---

## 15. Comparison với Current Architecture

| Aspect | Current CA-SAMoE | Proposed Hierarchical |
|--------|------------------|----------------------|
| Routing granularity | Utterance-level | Token + Cluster + Utterance |
| Router complexity | 1 Linear | Multi-level MLP/Attention |
| Load balancing | Single (final gates) | Multi-level (token/cluster/group) |
| Expert grouping | None | Static/Dynamic groups |
| Cluster formation | N/A | Length/Similarity/Window |
| Evolution speed | 1 expert/epoch | Multiple experts/epoch |
| Specialization | Emergent | Encouraged via groups |
| Parameters | Baseline | +10-20% routers |
| Compute overhead | Low (N+1 forward) | Medium (clustering cost) |
| Flexibility | Medium | High |

---

## 16. Conclusion

Kiến trúc mới đề xuất:

1. ✅ **Hybrid routing**: Token-level để capture local patterns, cluster-level để reduce noise, utterance-level cho global coherence
2. ✅ **Expert groups**: Specialization được khuyến khích thay vì để tự nhiên
3. ✅ **Multi-level load balance**: Comprehensive monitoring và balancing ở mọi granularity
4. ✅ **Improved evolution**: Clonal selection + speciation cho đa dạng nhanh hơn
5. ✅ **Advanced routers**: Hierarchical/Attention thay vì single Linear

**Trade-offs**:
- **Complexity tăng**: More components, harder to debug
- **Compute tăng**: Clustering cost, nhưng có optimize
- **Training time tăng**: More hyperparameters, cần nhiều ablation

**Expected outcome**: 
- Better accuracy (especially trên diverse datasets)
- More interpretable expert specializations
- Faster convergence (vì better gradient signal từ multi-level)
- Potentially better scaling với nhiều experts

**Next steps**:
1. Implement Phase 1-2 (core components)
2. Test on small dataset (Indic/Telugu 10h)
3. Run ablations so sánh với baseline
4. Iterate dựa trên results

---

**File**: `ARCHITECTURE_REDESIGN.md`  
**Date**: 2026-04-28  
**Based on**: Diagram analysis + Current CA-SAMoE code review
