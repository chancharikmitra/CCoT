import json
import argparse

def filter_by_category(input_file, output_file, filter_list):

    category_counters = {}

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            obj = json.loads(line)
            category = obj.get('category')
            
         
            category_counters[category] = category_counters.get(category, 0) + 1
            
            if category in filter_list:
                if category_counters[category] % 1 == 0:
                    f_out.write(json.dumps(obj))
                    f_out.write('\n')

def main():
    filter_list = ["Instances Counting", "Scene Understanding","Instance Identity", "Instance Attributes", "Instance Location", "Spatial Relation", "Visual Reasoning", "Text Understanding", "Instance Interaction"] 
    filter_by_category("llava-seed-bench.jsonl", "llava-seed-bench-filtered.jsonl", filter_list)

if __name__ == "__main__":
    main()
