import json
import os

def fix_json_counting(input_file, output_file):
    """
    Fix inconsistent step and global_batch counting in concatenated JSON logs.
    Creates continuous counters that don't reset from checkpoint resumptions.
    """
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    fixed_events = []
    continuous_step = 0
    continuous_global_batch = 0
    JSON_LOG_STEPS = 50  # Your logging frequency
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                event = json.loads(line.strip())
                
                # Fix batch events
                if event.get('event_type') == 'batch':
                    continuous_step += JSON_LOG_STEPS  # Increment by actual training steps
                    continuous_global_batch += JSON_LOG_STEPS
                    
                    # Update the event with continuous counters
                    event['step'] = continuous_step
                    if 'metrics' in event and 'global_batch' in event['metrics']:
                        event['metrics']['global_batch'] = continuous_global_batch
                
                # Fix epoch_end events  
                elif event.get('event_type') == 'epoch_end':
                    # Use current continuous counters
                    event['step'] = continuous_step
                    if 'metrics' in event and 'global_batch' in event['metrics']:
                        event['metrics']['global_batch'] = continuous_global_batch
                
                fixed_events.append(event)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    # Write fixed events
    with open(output_file, 'w') as f:
        for event in fixed_events:
            f.write(json.dumps(event) + '\n')
    
    print(f"Fixed {len(fixed_events)} events")
    print(f"Final continuous step: {continuous_step} (accounting for JSON_LOG_STEPS={JSON_LOG_STEPS})")
    print(f"Final continuous global_batch: {continuous_global_batch}")

def main():
    # Configure your file paths
    input_file = './outputs/sym_4gpu_final/logs/symbolic_4gpu_final_20250610_091603.jsonl'  # Your combined file
    output_file = './outputs/sym_4gpu_final/logs/symbolic_4gpu_final_fixed.jsonl'    # Fixed output
    
    # Check if input exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Please update the input_file path to match your combined JSON file")
        return
    
    # Create backup
    backup_file = input_file + '.backup'
    if not os.path.exists(backup_file):
        print(f"Creating backup: {backup_file}")
        with open(input_file, 'r') as src, open(backup_file, 'w') as dst:
            dst.write(src.read())
    
    # Fix the counting
    fix_json_counting(input_file, output_file)
    
    print(f"\nDone! Use the fixed file: {output_file}")
    print("You can now run your plotting script on the fixed file.")

if __name__ == "__main__":
    main()