#!/usr/bin/env python3
"""
Run comprehensive attention analysis with detailed "Other" attention breakdown.

Usage:
    python run_attention_analysis_detailed.py --checkpoint path/to/checkpoint.pt --model-type vanilla --output-prefix analysis

This version explicitly shows what tokens are classified as "Other" attention.
"""

import sys
import os
sys.path.append('/Users/alexisfox/st')

import argparse
import json
from datetime import datetime
import numpy as np
from attention import load_model_from_checkpoint, load_tokenizer, AttentionSample

def analyze_sentence_level_attention(model, tokenizer, sample, device='cpu'):
    """Aggregate attention by sentence when predicting answer"""
    from src.hooks.inference import AttentionExtractionHook
    from src.inference.generation import run_generation
    
    att_sample = AttentionSample.from_content(sample)
    context = ' '.join(att_sample.sentences) + f' {att_sample.question} Answer:'
    
    attention_hook = AttentionExtractionHook(threshold=0.0, store_values=False, tokenizer=tokenizer)
    
    try:
        ids, generated_text = run_generation(
            model=model, tokenizer=tokenizer, prompt_text=context,
            device=device, max_new_tokens=1, hooks=[attention_hook]
        )
        
        # Process attention from final prediction step
        token_attention = {}
        total_weight = 0
        
        # Aggregate attention across all heads for final prediction
        for record in attention_hook.attention_data[-6:]:  # Last 6 attention heads
            if 'attention_matrix' in record:
                attn_matrix = record['attention_matrix']
                tokens_in_record = record.get('tokens', [])
                
                if len(attn_matrix.shape) >= 2:
                    final_query_attention = attn_matrix[-1, :]
                    
                    for i, (token, weight) in enumerate(zip(tokens_in_record, final_query_attention)):
                        clean_token = str(token).replace('▁', '').replace('Ġ', '').strip()
                        if clean_token:
                            if clean_token not in token_attention:
                                token_attention[clean_token] = 0
                            token_attention[clean_token] += float(weight)
                            total_weight += float(weight)
        
        # Normalize attention weights
        if total_weight > 0:
            for token in token_attention:
                token_attention[token] = token_attention[token] / total_weight
        
        # Map tokens to sentences and categorize "Other"
        sentence_attention = [0.0] * len(att_sample.sentences)
        question_attention = 0.0
        other_attention = 0.0
        other_tokens_detail = {}  # Track individual other tokens
        
        # Create word-to-sentence mapping
        sentence_words = []
        for i, sentence in enumerate(att_sample.sentences):
            words = sentence.lower().split()
            sentence_words.append((i, words))
        
        question_words = att_sample.question.lower().split()
        
        # Aggregate attention by sentence and track "Other" tokens
        for token, attention_weight in token_attention.items():
            token_lower = token.lower()
            
            # Check which sentence this token belongs to
            matched_sentence = False
            
            # First check exact matches in sentences
            for sentence_idx, words in sentence_words:
                if token_lower in words or any(token_lower in word for word in words):
                    sentence_attention[sentence_idx] += attention_weight
                    matched_sentence = True
                    break
            
            # If not matched, check if it's from the question
            if not matched_sentence:
                if token_lower in question_words or any(token_lower in word for word in question_words):
                    question_attention += attention_weight
                    matched_sentence = True
            
            # If still not matched, it's "Other" - track the specific token
            if not matched_sentence:
                other_attention += attention_weight
                other_tokens_detail[token] = attention_weight
        
        return {
            'sample_id': att_sample.id,
            'question': att_sample.question,
            'answer': att_sample.answer,
            'sentences': att_sample.sentences,
            'generated_answer': generated_text.split()[-1] if generated_text else 'None',
            'sentence_attention': sentence_attention,
            'question_attention': question_attention,
            'other_attention': other_attention,
            'other_tokens_detail': other_tokens_detail,  # NEW: specific other tokens
            'token_attention': token_attention
        }
        
    except Exception as e:
        return {'sample_id': att_sample.id, 'error': str(e)}

def categorize_other_token(token):
    """Categorize what type of 'Other' token this is"""
    if token == ':':
        return 'colon punctuation'
    elif token == '?':
        return 'question mark'
    elif token == '.':
        return 'period'
    elif token.lower() == 'answer':
        return 'prompt word'
    elif token in ['"', "'", '(', ')', '[', ']', '{', '}']:
        return 'punctuation'
    elif len(token.strip()) == 0:
        return 'whitespace'
    elif token.startswith('▁') or token.startswith('Ġ'):
        return 'tokenizer artifact'
    else:
        return 'other formatting'

def create_detailed_analysis_report(results, checkpoint_path, model_type, output_prefix):
    """Create comprehensive analysis report with detailed Other breakdown"""
    
    report_lines = []
    report_lines.append('=' * 100)
    report_lines.append('📊 DETAILED SENTENCE-LEVEL ATTENTION ANALYSIS')
    report_lines.append('=' * 100)
    report_lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    report_lines.append(f'Checkpoint: {checkpoint_path}')
    report_lines.append(f'Model Type: {model_type}')
    report_lines.append('Analysis: Attention when predicting word after "Answer:" with detailed "Other" breakdown')
    report_lines.append('')
    
    correct_predictions = 0
    total_samples = 0
    avg_causal_attention = 0
    avg_fact_attention = 0
    avg_question_attention = 0
    avg_other_attention = 0
    
    # Track Other token types across all samples
    all_other_tokens = {}
    
    for result in results:
        if 'error' not in result:
            total_samples += 1
            
            report_lines.append('─' * 100)
            report_lines.append(f'📋 SAMPLE {result["sample_id"]}: {result["question"]}')
            report_lines.append('─' * 100)
            report_lines.append(f'Expected: {result["answer"]} | Generated: {result["generated_answer"]}')
            
            # Check correctness
            is_correct = (result['generated_answer'].lower() in ['yes', 'no'] and 
                         result['generated_answer'].lower() == result['answer'].lower())
            if is_correct:
                correct_predictions += 1
                status = '✅ CORRECT'
            else:
                status = '❌ WRONG'
            report_lines.append(f'Status: {status}')
            report_lines.append('')
            
            # Sentence breakdown
            report_lines.append('📝 SENTENCE ATTENTION:')
            sentence_scores = result['sentence_attention']
            max_idx = np.argmax(sentence_scores) if sentence_scores else 0
            
            causal_attention = 0
            fact_attention = 0
            
            for j, (sentence, attention) in enumerate(zip(result['sentences'], sentence_scores)):
                is_causal = any(word in sentence.lower() for word in ['when', 'if', 'because', 'need', "haven't"])
                sentence_type = '🎯 CAUSAL' if is_causal else '📋 FACT'
                attended = '👁️' if j == max_idx else '  '
                
                bar_length = int(attention * 50)
                bar = '█' * bar_length + '░' * max(0, 50 - bar_length)
                
                report_lines.append(f'  {sentence_type} {attended} [{j+1}] {attention:.3f} [{bar}]')
                report_lines.append(f'      {sentence}')
                
                if is_causal:
                    causal_attention += attention
                else:
                    fact_attention += attention
            
            # NEW: Detailed "Other" attention breakdown
            report_lines.append('')
            report_lines.append('🔍 "OTHER" ATTENTION BREAKDOWN (Non-content tokens):')
            if 'other_tokens_detail' in result:
                other_tokens = sorted(result['other_tokens_detail'].items(), key=lambda x: x[1], reverse=True)
                
                for token, weight in other_tokens[:5]:  # Top 5 other tokens
                    token_type = categorize_other_token(token)
                    pct_total = weight * 100
                    pct_other = (weight / result['other_attention'] * 100) if result['other_attention'] > 0 else 0
                    
                    report_lines.append(f'    "{token}" ({token_type}): {weight:.4f} ({pct_total:.1f}% total, {pct_other:.1f}% of Other)')
                    
                    # Track for overall stats
                    if token not in all_other_tokens:
                        all_other_tokens[token] = []
                    all_other_tokens[token].append(weight)
                
                if len(other_tokens) > 5:
                    report_lines.append(f'    ... and {len(other_tokens)-5} more other tokens')
            
            # Statistics
            total_content = causal_attention + fact_attention
            causal_pct = (causal_attention / total_content * 100) if total_content > 0 else 0
            
            report_lines.append('')
            report_lines.append(f'📊 ATTENTION BREAKDOWN:')
            report_lines.append(f'    🎯 Causal: {causal_attention:.3f} ({causal_pct:.1f}%)')
            report_lines.append(f'    📋 Facts: {fact_attention:.3f} ({100-causal_pct:.1f}%)')
            report_lines.append(f'    ❓ Question: {result["question_attention"]:.3f}')
            report_lines.append(f'    📋 Other: {result["other_attention"]:.3f} ({result["other_attention"]*100:.1f}%)')
            
            # Assessment
            if causal_pct > 60:
                assessment = '✅ EXCELLENT - Strong causal focus'
            elif causal_pct > 40:
                assessment = '⚠️  GOOD - Moderate causal focus'
            elif causal_pct > 20:
                assessment = '❌ POOR - Weak causal focus'
            else:
                assessment = '💀 TERRIBLE - No causal focus'
            
            report_lines.append(f'    Assessment: {assessment}')
            
            # Show biggest attention waste
            if 'other_tokens_detail' in result and result['other_tokens_detail']:
                biggest_waste = max(result['other_tokens_detail'].items(), key=lambda x: x[1])
                report_lines.append(f'    💀 Biggest waste: "{biggest_waste[0]}" gets {biggest_waste[1]*100:.1f}% of total attention')
            
            report_lines.append('')
            
            # Accumulate averages
            avg_causal_attention += causal_attention
            avg_fact_attention += fact_attention
            avg_question_attention += result['question_attention']
            avg_other_attention += result['other_attention']
    
    # Overall stats with detailed Other breakdown
    if total_samples > 0:
        avg_causal_attention /= total_samples
        avg_fact_attention /= total_samples
        avg_question_attention /= total_samples
        avg_other_attention /= total_samples
    
    accuracy = (correct_predictions / total_samples * 100) if total_samples > 0 else 0
    total_content_avg = avg_causal_attention + avg_fact_attention
    causal_pct_avg = (avg_causal_attention / total_content_avg * 100) if total_content_avg > 0 else 0
    
    report_lines.append('=' * 100)
    report_lines.append('📈 OVERALL STATISTICS WITH DETAILED "OTHER" ANALYSIS')
    report_lines.append('=' * 100)
    report_lines.append(f'🎯 Answer Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_samples})')
    report_lines.append(f'📊 Average Attention:')
    report_lines.append(f'    🎯 Causal sentences: {avg_causal_attention:.3f} ({causal_pct_avg:.1f}%)')
    report_lines.append(f'    📋 Fact sentences: {avg_fact_attention:.3f} ({100-causal_pct_avg:.1f}%)')
    report_lines.append(f'    ❓ Question: {avg_question_attention:.3f}')
    report_lines.append(f'    📋 Other: {avg_other_attention:.3f} ({avg_other_attention*100:.1f}%)')
    report_lines.append('')
    
    # Detailed Other token analysis
    report_lines.append('🔍 DETAILED "OTHER" TOKEN BREAKDOWN:')
    report_lines.append('Average attention per token type across all samples:')
    
    avg_other_by_token = {}
    for token, weights in all_other_tokens.items():
        avg_other_by_token[token] = np.mean(weights)
    
    sorted_other = sorted(avg_other_by_token.items(), key=lambda x: x[1], reverse=True)
    
    for token, avg_weight in sorted_other[:10]:
        token_type = categorize_other_token(token)
        pct_total = avg_weight * 100
        report_lines.append(f'    "{token}" ({token_type}): {avg_weight:.4f} ({pct_total:.1f}% average)')
    
    report_lines.append('')
    report_lines.append('🔍 KEY FINDINGS:')
    if accuracy < 10:
        report_lines.append('   • ❌ CRITICAL: Model cannot generate proper Yes/No answers')
    if causal_pct_avg < 30:
        report_lines.append('   • ❌ POOR: Insufficient attention to causal reasoning sentences')
    if avg_other_attention > 0.5:
        report_lines.append('   • ❌ CRITICAL: Massive attention waste on formatting (>50%)')
    elif avg_other_attention > 0.3:
        report_lines.append('   • ❌ POOR: Too much attention wasted on non-content words (>30%)')
    
    # Identify biggest offender
    if sorted_other:
        biggest_offender = sorted_other[0]
        report_lines.append(f'   • 💀 BIGGEST PROBLEM: "{biggest_offender[0]}" wastes {biggest_offender[1]*100:.1f}% of attention on average')
    
    if avg_other_attention > 0.1:
        report_lines.append('')
        report_lines.append('💡 RECOMMENDATION: Reduce "Other" attention to <10% for efficient reasoning')
        report_lines.append('Current model wastes too much attention on formatting instead of content')
    
    return '\\n'.join(report_lines)

def main():
    parser = argparse.ArgumentParser(description='Run detailed attention analysis with Other token breakdown')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='vanilla', choices=['vanilla', 'symbolic', 'tft'],
                       help='Model architecture type')
    parser.add_argument('--data-path', type=str, default='attention_data.json', 
                       help='Path to attention test data')
    parser.add_argument('--output-prefix', type=str, default='detailed_analysis',
                       help='Prefix for output files')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on')
    parser.add_argument('--samples', type=int, default=-1, help='Number of samples to analyze (-1 = all)')
    
    args = parser.parse_args()
    
    print(f'🎯 RUNNING DETAILED ATTENTION ANALYSIS')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Model Type: {args.model_type}')
    print(f'Device: {args.device}')
    print('=' * 60)
    
    # Load model and tokenizer
    print('Loading model and tokenizer...')
    tokenizer, _ = load_tokenizer('gpt2')
    model, _, _ = load_model_from_checkpoint(args.checkpoint, args.device, args.model_type)
    
    # Load test data
    print('Loading test data...')
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    samples_to_analyze = data['samples']
    if args.samples > 0:
        samples_to_analyze = samples_to_analyze[:args.samples]
    
    print(f'Analyzing {len(samples_to_analyze)} samples...')
    
    # Run analysis
    results = []
    for i, sample_data in enumerate(samples_to_analyze):
        print(f'Processing sample {i+1}/{len(samples_to_analyze)}...', end=' ')
        result = analyze_sentence_level_attention(model, tokenizer, sample_data, args.device)
        results.append(result)
        
        if 'error' in result:
            print(f'ERROR: {result["error"]}')
        else:
            print(f'Generated: {result["generated_answer"]}')
    
    # Create detailed report
    print('\\nGenerating detailed report...')
    
    detailed_report = create_detailed_analysis_report(results, args.checkpoint, args.model_type, args.output_prefix)
    report_file = f'{args.output_prefix}_detailed_attention_analysis.txt'
    with open(report_file, 'w') as f:
        f.write(detailed_report)
    
    # Raw data
    data_file = f'{args.output_prefix}_detailed_attention_data.json'
    with open(data_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print('\\n✅ Detailed analysis complete!')
    print(f'📄 Report: {report_file}')
    print(f'📊 Data: {data_file}')
    
    # Show quick summary
    valid_results = [r for r in results if 'error' not in r]
    correct = sum(1 for r in valid_results if r['generated_answer'].lower() in ['yes', 'no'] and 
                  r['generated_answer'].lower() == r['answer'].lower())
    total = len(valid_results)
    accuracy = (correct/total*100) if total > 0 else 0
    avg_other = np.mean([r['other_attention'] for r in valid_results]) if valid_results else 0
    
    print(f'\\n🎯 QUICK SUMMARY:')
    print(f'   Answer Accuracy: {accuracy:.1f}% ({correct}/{total})')
    print(f'   Average "Other" Attention: {avg_other*100:.1f}%')
    print(f'   Samples Analyzed: {total}')

if __name__ == '__main__':
    main()