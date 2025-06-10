import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

class ImprovedSymbolicKGExtractor:
    """
    Improved version with better analysis and visualization.
    """
    
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = model.config.vocab_size
        self.n_head = model.config.n_head
        self.n_layer = model.config.n_layer
        self.vocab_weights = []
        self.attention_weights = []
        
        # Create a simple token mapping for display
        self.token_map = {i: f"tok_{i}" for i in range(self.vocab_size)}
        
    def extract_detailed_vocab_weights(self, input_ids):
        """Extract vocabulary weights with detailed analysis."""
        self.vocab_weights = []
        layer_stats = []
        
        def vocab_hook(module, input, output):
            if hasattr(module, 'channel_vocab_attentions'):
                x = input[0]
                B, T, C = x.shape
                x_channels = x.view(B, T, module.n_head, module.head_dim)
                
                layer_weights = []
                layer_entropies = []
                layer_top_tokens = []
                
                for h in range(module.n_head):
                    x_h = x_channels[:, :, h, :]
                    ffn_h = module.channel_ffns[h](x_h)
                    vocab_logits_h = module.channel_vocab_attentions[h](ffn_h)
                    temp_h = torch.clamp(module.channel_temperatures[h], min=0.1)
                    vocab_weights_h = F.softmax(vocab_logits_h / temp_h, dim=-1)
                    
                    layer_weights.append(vocab_weights_h.detach())
                    
                    # Calculate entropy (measure of distribution sharpness)
                    entropy = -torch.sum(vocab_weights_h * torch.log(vocab_weights_h + 1e-10), dim=-1)
                    layer_entropies.append(entropy.mean().item())
                    
                    # Find most active tokens
                    top_weights, top_indices = torch.topk(vocab_weights_h.mean(dim=(0,1)), k=10)
                    layer_top_tokens.append(list(zip(top_indices.tolist(), top_weights.tolist())))
                
                self.vocab_weights.append(layer_weights)
                layer_stats.append({
                    'entropies': layer_entropies,
                    'top_tokens': layer_top_tokens,
                    'temp_values': [module.channel_temperatures[h].item() for h in range(module.n_head)]
                })
        
        # Register hooks
        handles = []
        for layer in self.model.transformer.h:
            if hasattr(layer, 'ffn'):
                handle = layer.ffn.register_forward_hook(vocab_hook)
                handles.append(handle)
        
        with torch.no_grad():
            output = self.model(input_ids)
        
        for handle in handles:
            handle.remove()
        
        return layer_stats
    
    def analyze_layer_evolution(self):
        """Analyze how token representations evolve across layers."""
        if not self.vocab_weights:
            raise ValueError("Must extract weights first")
        
        print("\n=== LAYER-BY-LAYER EVOLUTION ===")
        
        for layer_idx, layer_weights in enumerate(self.vocab_weights):
            print(f"\nLayer {layer_idx}:")
            
            for head_idx, head_weights in enumerate(layer_weights):
                B, T, V = head_weights.shape
                
                # Calculate statistics
                mean_weights = head_weights.mean(dim=(0,1))  # Average across batch and time
                max_weight = mean_weights.max().item()
                entropy = -torch.sum(mean_weights * torch.log(mean_weights + 1e-10)).item()
                
                # Find top tokens
                top_k = 5
                top_values, top_indices = torch.topk(mean_weights, top_k)
                
                print(f"  Head {head_idx}: max_weight={max_weight:.3f}, entropy={entropy:.2f}")
                print(f"    Top tokens: {[(idx.item(), val.item()) for idx, val in zip(top_indices, top_values)]}")
    
    def build_semantic_graph(self, threshold=0.05, min_weight=0.01):
        """Build graph with better semantic analysis."""
        if not self.vocab_weights:
            raise ValueError("Must extract weights first")
        
        G = nx.Graph()
        token_activations = defaultdict(float)  # Track how often each token is active
        
        for layer_idx, layer_weights in enumerate(self.vocab_weights):
            for head_idx, head_weights in enumerate(layer_weights):
                B, T, V = head_weights.shape
                
                for batch in range(B):
                    for pos in range(T):
                        weights = head_weights[batch, pos]
                        
                        # Track token activations
                        for token_id, weight in enumerate(weights):
                            if weight > min_weight:
                                token_activations[token_id] += weight.item()
                        
                        # Find co-occurring tokens
                        active_tokens = torch.where(weights > threshold)[0]
                        
                        if len(active_tokens) > 1:
                            for i, token_i in enumerate(active_tokens):
                                for token_j in active_tokens[i+1:]:
                                    weight_i = weights[token_i].item()
                                    weight_j = weights[token_j].item()
                                    edge_weight = min(weight_i, weight_j)
                                    
                                    if G.has_edge(token_i.item(), token_j.item()):
                                        G[token_i.item()][token_j.item()]['weight'] += edge_weight
                                        G[token_i.item()][token_j.item()]['count'] += 1
                                    else:
                                        G.add_edge(token_i.item(), token_j.item(), 
                                                 weight=edge_weight, count=1,
                                                 layer=layer_idx, head=head_idx)
        
        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['activation'] = token_activations[node]
            G.nodes[node]['token_name'] = self.token_map[node]
        
        return G, token_activations
    
    def find_token_clusters(self, G, min_cluster_size=3):
        """Find clusters using graph community detection."""
        if G.number_of_edges() == 0:
            return []
        
        # Use simple connected components as clusters
        clusters = []
        for component in nx.connected_components(G):
            if len(component) >= min_cluster_size:
                # Calculate cluster strength
                subgraph = G.subgraph(component)
                total_weight = sum([data['weight'] for _, _, data in subgraph.edges(data=True)])
                clusters.append({
                    'tokens': list(component),
                    'size': len(component),
                    'weight': total_weight,
                    'density': subgraph.number_of_edges() / (len(component) * (len(component) - 1) / 2)
                })
        
        # Sort by cluster strength
        clusters.sort(key=lambda x: x['weight'], reverse=True)
        return clusters
    
    def visualize_top_relationships(self, G, top_k=20):
        """Show the strongest token relationships."""
        if G.number_of_edges() == 0:
            print("No edges in graph to visualize")
            return
        
        # Get top edges by weight
        edges_by_weight = [(u, v, data['weight']) for u, v, data in G.edges(data=True)]
        edges_by_weight.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\n=== TOP {top_k} TOKEN RELATIONSHIPS ===")
        for i, (token1, token2, weight) in enumerate(edges_by_weight[:top_k]):
            count = G[token1][token2]['count']
            print(f"{i+1:2d}. Token {token1} <-> Token {token2}: weight={weight:.4f}, count={count}")
    
    def analyze_token_importance(self, token_activations, top_k=20):
        """Analyze which tokens are most important overall."""
        sorted_tokens = sorted(token_activations.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n=== TOP {top_k} MOST ACTIVE TOKENS ===")
        for i, (token_id, activation) in enumerate(sorted_tokens[:top_k]):
            print(f"{i+1:2d}. Token {token_id}: activation={activation:.4f}")
    
    def comprehensive_analysis(self, input_ids):
        """Run complete analysis."""
        print("🔍 COMPREHENSIVE SYMBOLIC TRANSFORMER ANALYSIS")
        print("=" * 60)
        
        # Extract weights with detailed stats
        layer_stats = self.extract_detailed_vocab_weights(input_ids)
        
        # Analyze layer evolution
        self.analyze_layer_evolution()
        
        # Build semantic graph
        G, token_activations = self.build_semantic_graph(threshold=0.05)
        
        print(f"\n=== GRAPH STATISTICS ===")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")
        print(f"Avg degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
        
        # Find clusters
        clusters = self.find_token_clusters(G)
        print(f"\n=== FOUND {len(clusters)} CLUSTERS ===")
        for i, cluster in enumerate(clusters[:5]):  # Show top 5
            print(f"Cluster {i+1}: {cluster['size']} tokens, weight={cluster['weight']:.3f}, density={cluster['density']:.3f}")
            print(f"  Tokens: {cluster['tokens'][:10]}...")  # Show first 10 tokens
        
        # Show top relationships
        self.visualize_top_relationships(G)
        
        # Show most important tokens
        self.analyze_token_importance(token_activations)
        
        return {
            'graph': G,
            'clusters': clusters,
            'token_activations': token_activations,
            'layer_stats': layer_stats
        }

def run_improved_analysis(model):
    """Run the improved analysis."""
    extractor = ImprovedSymbolicKGExtractor(model)
    
    # Create sample inputs
    sample_inputs = [torch.randint(0, model.config.vocab_size, (1, 15)) for _ in range(2)]
    
    all_results = []
    
    for i, input_ids in enumerate(sample_inputs):
        print(f"\n{'='*60}")
        print(f"ANALYZING SAMPLE {i+1}")
        print(f"{'='*60}")
        
        results = extractor.comprehensive_analysis(input_ids)
        all_results.append(results)
    
    return extractor, all_results

# Usage
if __name__ == "__main__":
    # Load your model (using the same loading code as before)
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Same loading logic as before...
    first_key = list(checkpoint.keys())[0] if checkpoint else ""
    
    if first_key.startswith('module.'):
        model_state_dict = checkpoint
        
        if 'module.transformer.wte.weight' in model_state_dict:
            n_embd = model_state_dict['module.transformer.wte.weight'].shape[1]
            vocab_size = model_state_dict['module.transformer.wte.weight'].shape[0]
        elif 'transformer.wte.weight' in model_state_dict:
            n_embd = model_state_dict['transformer.wte.weight'].shape[1] 
            vocab_size = model_state_dict['transformer.wte.weight'].shape[0]
        
        if n_embd == 384:
            preset = 'medium'
        elif n_embd == 192:
            preset = 'small'
        else:
            preset = 'medium'
        
        config = get_preset_config(preset)
        config.vocab_size = vocab_size
        config.n_embd = n_embd
        
        fixed_state_dict = {}
        for key, value in model_state_dict.items():
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            fixed_state_dict[new_key] = value
        
        model_state_dict = fixed_state_dict
    
    model = get_model("Symbolic", config=config)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Run improved analysis
    extractor, results = run_improved_analysis(model)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("✅ Vocabulary mixture weights extracted and analyzed")
    print("✅ Token relationships and clusters identified")
    print("✅ Layer evolution patterns analyzed")