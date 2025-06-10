import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from collections import defaultdict
from pathlib import Path

class SymbolicKnowledgeGraphExtractor:
    """
    Extract knowledge graphs from Symbolic Transformer internal representations.
    
    The key insight: Every hidden state x can be written as:
    x = Σ_{h=1}^H Σ_{i=1}^{|V|} w_{h,i} E_{i,h}
    
    The vocab projection FFN gives us these weights w_{h,i} directly!
    """
    
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = model.config.vocab_size
        self.n_head = model.config.n_head
        self.n_layer = model.config.n_layer
        
        # Storage for extracted weights
        self.vocab_weights = []  # List of [layer][head] -> (B, T, vocab_size)
        self.attention_weights = []  # List of attention patterns
        
    def extract_vocab_weights(self, input_ids, return_attention=True):
        """
        Extract vocabulary mixture weights from all layers.
        These weights show which vocab tokens are 'active' at each position.
        """
        self.vocab_weights = []
        self.attention_weights = []
        
        # Hook to capture vocabulary attention weights from VocabularyProjectionFFN
        def vocab_hook(module, input, output):
            if hasattr(module, 'channel_vocab_attentions'):
                # Extract the vocab weights computed during forward pass
                x = input[0]  # (B, T, n_embd)
                B, T, C = x.shape
                x_channels = x.view(B, T, module.n_head, module.head_dim)
                
                layer_weights = []
                for h in range(module.n_head):
                    x_h = x_channels[:, :, h, :]
                    ffn_h = module.channel_ffns[h](x_h)
                    vocab_logits_h = module.channel_vocab_attentions[h](ffn_h)
                    temp_h = torch.clamp(module.channel_temperatures[h], min=0.1)
                    vocab_weights_h = F.softmax(vocab_logits_h / temp_h, dim=-1)
                    layer_weights.append(vocab_weights_h.detach())
                
                self.vocab_weights.append(layer_weights)
        
        # Hook to capture attention patterns
        def attention_hook(module, input, output):
            if hasattr(module, 'attn_weights') and module.attn_weights is not None:
                self.attention_weights.append(module.attn_weights.detach())
        
        # Register hooks
        vocab_handles = []
        attn_handles = []
        
        for layer_idx, layer in enumerate(self.model.transformer.h):
            if hasattr(layer, 'ffn'):
                handle = layer.ffn.register_forward_hook(vocab_hook)
                vocab_handles.append(handle)
            
            if return_attention and hasattr(layer, 'attn'):
                handle = layer.attn.register_forward_hook(attention_hook)
                attn_handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_ids)
        
        # Remove hooks
        for handle in vocab_handles + attn_handles:
            handle.remove()
        
        return self.vocab_weights, self.attention_weights
    
    def build_token_cooccurrence_graph(self, threshold=0.1, layer=None):
        """
        Build graph where edges connect tokens that co-occur with high weights
        at the same positions. This captures conceptual relationships.
        """
        if not self.vocab_weights:
            raise ValueError("Must call extract_vocab_weights first")
        
        G = nx.Graph()
        
        # Select layers to analyze
        layers_to_analyze = [layer] if layer is not None else range(len(self.vocab_weights))
        
        for layer_idx in layers_to_analyze:
            layer_weights = self.vocab_weights[layer_idx]  # List of head weights
            
            for head_idx, head_weights in enumerate(layer_weights):
                B, T, V = head_weights.shape
                
                for batch in range(B):
                    for pos in range(T):
                        # Get active tokens at this position (above threshold)
                        weights = head_weights[batch, pos]  # (vocab_size,)
                        active_tokens = torch.where(weights > threshold)[0]
                        
                        # Add edges between co-occurring tokens
                        for i, token_i in enumerate(active_tokens):
                            for token_j in active_tokens[i+1:]:
                                weight = min(weights[token_i], weights[token_j]).item()
                                
                                if G.has_edge(token_i.item(), token_j.item()):
                                    G[token_i.item()][token_j.item()]['weight'] += weight
                                else:
                                    G.add_edge(token_i.item(), token_j.item(), 
                                             weight=weight, 
                                             layer=layer_idx, 
                                             head=head_idx)
        
        return G
    
    def build_attention_flow_graph(self, threshold=0.1):
        """
        Build graph where edges represent attention flow between tokens.
        This captures which tokens the model thinks are related.
        """
        if not self.attention_weights:
            raise ValueError("Must call extract_vocab_weights with return_attention=True first")
        
        G = nx.DiGraph()  # Directed graph for attention flow
        
        for layer_idx, attn_weights in enumerate(self.attention_weights):
            B, H, T, T_key = attn_weights.shape
            
            # For each head
            for head in range(H):
                for batch in range(B):
                    attn_matrix = attn_weights[batch, head]  # (T, T)
                    
                    # Find high-attention connections
                    high_attn = torch.where(attn_matrix > threshold)
                    
                    for i, j in zip(high_attn[0], high_attn[1]):
                        if i != j:  # Skip self-attention
                            weight = attn_matrix[i, j].item()
                            
                            # Note: In real implementation, you'd want to map positions back to tokens
                            # For now, using position indices
                            if G.has_edge(i.item(), j.item()):
                                G[i.item()][j.item()]['weight'] += weight
                            else:
                                G.add_edge(i.item(), j.item(), 
                                         weight=weight, 
                                         layer=layer_idx, 
                                         head=head)
        
        return G
    
    def extract_layer_abstractions(self, text_input, top_k=10):
        """
        Show how token representations evolve across layers.
        Higher layers should show more abstract relationships.
        """
        # Tokenize input (you'll need to implement this with your tokenizer)
        if isinstance(text_input, str):
            # For demo, using random tokens - replace with actual tokenization
            input_ids = torch.randint(0, self.vocab_size, (1, len(text_input.split())))
        else:
            input_ids = text_input
        
        vocab_weights, _ = self.extract_vocab_weights(input_ids, return_attention=False)
        
        abstractions = {}
        
        for layer_idx, layer_weights in enumerate(vocab_weights):
            layer_abstractions = []
            
            for head_idx, head_weights in enumerate(layer_weights):
                B, T, V = head_weights.shape
                
                # For each position, find top-k active tokens
                for pos in range(T):
                    weights = head_weights[0, pos]  # Assuming batch size 1
                    top_indices = torch.topk(weights, top_k).indices
                    top_weights = weights[top_indices]
                    
                    active_tokens = [(idx.item(), weight.item()) 
                                   for idx, weight in zip(top_indices, top_weights)]
                    
                    layer_abstractions.append({
                        'position': pos,
                        'head': head_idx,
                        'active_tokens': active_tokens
                    })
            
            abstractions[f'layer_{layer_idx}'] = layer_abstractions
        
        return abstractions
    
    def find_semantic_clusters(self, threshold=0.15, min_cluster_size=3):
        """
        Find clusters of tokens that frequently co-occur across positions.
        These likely represent semantic concepts.
        """
        if not self.vocab_weights:
            raise ValueError("Must call extract_vocab_weights first")
        
        # Count co-occurrences across all positions and layers
        cooccurrence_counts = defaultdict(int)
        
        for layer_weights in self.vocab_weights:
            for head_weights in layer_weights:
                B, T, V = head_weights.shape
                
                for batch in range(B):
                    for pos in range(T):
                        weights = head_weights[batch, pos]
                        active_tokens = torch.where(weights > threshold)[0]
                        
                        # Count pairs
                        for i, token_i in enumerate(active_tokens):
                            for token_j in active_tokens[i+1:]:
                                pair = tuple(sorted([token_i.item(), token_j.item()]))
                                cooccurrence_counts[pair] += 1
        
        # Build similarity matrix
        all_tokens = set()
        for pair in cooccurrence_counts:
            all_tokens.update(pair)
        
        token_list = sorted(all_tokens)
        n_tokens = len(token_list)
        similarity_matrix = np.zeros((n_tokens, n_tokens))
        
        for (token_i, token_j), count in cooccurrence_counts.items():
            i = token_list.index(token_i)
            j = token_list.index(token_j)
            similarity_matrix[i, j] = count
            similarity_matrix[j, i] = count
        
        # Simple clustering (in practice, you'd use more sophisticated methods)
        clusters = []
        visited = set()
        
        for i, token in enumerate(token_list):
            if token in visited:
                continue
            
            # Find connected tokens
            cluster = [token]
            to_visit = [i]
            
            while to_visit:
                current_idx = to_visit.pop()
                current_token = token_list[current_idx]
                
                if current_token in visited:
                    continue
                visited.add(current_token)
                
                # Find similar tokens
                similarities = similarity_matrix[current_idx]
                similar_indices = np.where(similarities > min_cluster_size)[0]
                
                for sim_idx in similar_indices:
                    sim_token = token_list[sim_idx]
                    if sim_token not in visited and sim_token not in cluster:
                        cluster.append(sim_token)
                        to_visit.append(sim_idx)
            
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
        
        return clusters

def analyze_symbolic_transformer(model, text_samples=None):
    """
    Complete analysis of Symbolic Transformer knowledge representation.
    """
    extractor = SymbolicKnowledgeGraphExtractor(model)
    
    print("=== SYMBOLIC TRANSFORMER KNOWLEDGE GRAPH ANALYSIS ===\n")
    
    # Sample inputs for analysis
    if text_samples is None:
        # Use random token sequences for demo
        sample_inputs = [torch.randint(0, model.config.vocab_size, (1, 10)) for _ in range(3)]
    else:
        # Convert text to token ids (implement tokenization)
        sample_inputs = [torch.randint(0, model.config.vocab_size, (1, len(text.split()))) 
                        for text in text_samples]
    
    all_graphs = []
    
    for i, input_ids in enumerate(sample_inputs):
        print(f"--- Analyzing Sample {i+1} ---")
        
        # Extract vocabulary weights
        vocab_weights, attention_weights = extractor.extract_vocab_weights(input_ids)
        
        print(f"Extracted weights from {len(vocab_weights)} layers")
        print(f"Captured {len(attention_weights)} attention patterns")
        
        # Build co-occurrence graph
        cooccur_graph = extractor.build_token_cooccurrence_graph(threshold=0.1)
        print(f"Co-occurrence graph: {cooccur_graph.number_of_nodes()} nodes, {cooccur_graph.number_of_edges()} edges")
        
        # Build attention flow graph  
        if attention_weights:
            attention_graph = extractor.build_attention_flow_graph(threshold=0.1)
            print(f"Attention flow graph: {attention_graph.number_of_nodes()} nodes, {attention_graph.number_of_edges()} edges")
        
        # Extract layer abstractions
        abstractions = extractor.extract_layer_abstractions(input_ids)
        print(f"Layer abstractions extracted for {len(abstractions)} layers")
        
        all_graphs.append({
            'cooccurrence': cooccur_graph,
            'attention': attention_graph if attention_weights else None,
            'abstractions': abstractions
        })
        
        print()
    
    # Find semantic clusters across all samples
    print("--- Finding Semantic Clusters ---")
    clusters = extractor.find_semantic_clusters(threshold=0.1)
    print(f"Found {len(clusters)} semantic clusters:")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {cluster[:10]}...")  # Show first 10 tokens
    
    return extractor, all_graphs

# Usage example
if __name__ == "__main__":
    # Load your model using check.py logic
    import sys
    import os
    
    # Add parent directory for imports
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from config import get_preset_config
    from model import get_model
    
    checkpoint_path = "outputs/sym_4gpu_final/checkpoint_epoch_4.pt"
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Check checkpoint structure like check.py does
    first_key = list(checkpoint.keys())[0] if checkpoint else ""
    
    if first_key.startswith('module.'):
        # Checkpoint IS the model state dict
        print("✅ Checkpoint is model state dict (starts with 'module.')")
        model_state_dict = checkpoint
        
        # Auto-detect config from checkpoint weights
        if 'module.transformer.wte.weight' in model_state_dict:
            n_embd = model_state_dict['module.transformer.wte.weight'].shape[1]
            vocab_size = model_state_dict['module.transformer.wte.weight'].shape[0]
        elif 'transformer.wte.weight' in model_state_dict:
            n_embd = model_state_dict['transformer.wte.weight'].shape[1] 
            vocab_size = model_state_dict['transformer.wte.weight'].shape[0]
        else:
            print("❌ Could not find embedding weights to infer config")
            sys.exit(1)
        
        print(f"📊 Detected n_embd: {n_embd}, vocab_size: {vocab_size}")
        
        # Map to preset config
        if n_embd == 128:
            preset = 'tiny'
        elif n_embd == 192:
            preset = 'small'
        elif n_embd == 384:
            preset = 'medium'
        elif n_embd == 768:
            preset = 'large'
        else:
            print(f"Unknown embedding size {n_embd}, using medium as fallback")
            preset = 'medium'
        
        config = get_preset_config(preset)
        config.vocab_size = vocab_size
        config.n_embd = n_embd
        
        # Remove 'module.' prefix
        fixed_state_dict = {}
        for key, value in model_state_dict.items():
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            fixed_state_dict[new_key] = value
        
        model_state_dict = fixed_state_dict
        
    else:
        # Normal checkpoint format
        model_state_key = None
        for key in ['model_state_dict', 'model', 'state_dict']:
            if key in checkpoint:
                model_state_key = key
                break
        
        if model_state_key is None:
            print(f"❌ No model state dict found. Available keys: {list(checkpoint.keys())}")
            sys.exit(1)
        
        print(f"✅ Found model state at key: '{model_state_key}'")
        model_state_dict = checkpoint[model_state_key]
        
        # Get config from checkpoint if available
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Infer config from weights
            if 'transformer.wte.weight' in model_state_dict:
                n_embd = model_state_dict['transformer.wte.weight'].shape[1]
                vocab_size = model_state_dict['transformer.wte.weight'].shape[0]
                
                # Map to preset
                if n_embd == 384:
                    config = get_preset_config('medium')
                elif n_embd == 192:
                    config = get_preset_config('small')
                else:
                    config = get_preset_config('medium')
                
                config.vocab_size = vocab_size
                config.n_embd = n_embd
            else:
                print("❌ Could not infer config")
                sys.exit(1)
    
    # Create model
    is_symbolic = getattr(config, 'use_symbolic_ffn', True)
    if is_symbolic:
        model = get_model("Symbolic", config=config)
    else:
        model = get_model("Vanilla", config=config)
    
    # Load weights
    try:
        model.load_state_dict(model_state_dict)
        print("✅ Model loaded successfully")
    except RuntimeError as e:
        print(f"❌ Model loading error: {e}")
        sys.exit(1)
    
    model.eval()
    
    # Run analysis
    extractor, graphs = analyze_symbolic_transformer(model)
    
    print("\n=== KNOWLEDGE GRAPH EXTRACTION COMPLETE ===")
    print("✓ Vocabulary mixture weights extracted")
    print("✓ Token co-occurrence graphs built")
    print("✓ Attention flow patterns captured")
    print("✓ Layer-wise abstractions analyzed")
    print("✓ Semantic clusters identified")