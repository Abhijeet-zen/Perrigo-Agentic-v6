import re
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import uuid

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def get_prompt_file(data_source):
    """Return the appropriate prompt file based on the data source."""
    prompt_mapping = {
        'Outbound_Data.csv': 'prompt_templates/Prompt1.txt',
        'Inventory_Batch.csv': 'prompt_templates/Prompt2.txt',
        'Inbound_Data.csv': 'prompt_templates/Prompt3.txt'
    }
    return prompt_mapping.get(data_source)


def extract_code_segments(response_text):
    """Extract code segments from the API response using regex."""
    segments = {}

    # Extract approach section
    approach_match = re.search(r'<approach>(.*?)</approach>', response_text, re.DOTALL)
    if approach_match:
        segments['approach'] = approach_match.group(1).strip()

    # Extract content between <code> tags
    code_match = re.search(r'<code>(.*?)</code>', response_text, re.DOTALL)
    if code_match:
        segments['code'] = code_match.group(1).strip()

    # Extract content between <chart> tags
    chart_match = re.search(r'<chart>(.*?)</chart>', response_text, re.DOTALL)
    if chart_match:
        segments['chart'] = chart_match.group(1).strip()

    # Extract content between <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    if answer_match:
        segments['answer'] = answer_match.group(1).strip()

    return segments


def execute_codes(df, response_text):
    """Execute the extracted code segments on the provided dataframe and store formatted answer."""
    results = {
        'approach': None,
        'answer': None,
        'figure': None,
        'code': None,
        'chart_code': None
    }

    try:
        # Extract code segments
        # print("Insights Agent, Response Text\n",response_text)
        segments = extract_code_segments(response_text)
        if not segments:
            print("No code segments found in the response")
            return results

        # Store the approach and raw code
        if 'approach' in segments:
            results['approach'] = segments['approach']
        if 'code' in segments:
            results['code'] = segments['code']
        if 'chart' in segments:
            results['chart_code'] = segments['chart']

        # Create a single namespace for all executions
        namespace = {'df': df, 'pd': pd, 'plt': plt, 'sns': sns}

        # Execute analysis code and answer template
        if 'code' in segments and 'answer' in segments:
            # Properly dedent the code before execution
            code_lines = segments['code'].strip().split('\n')
            # Find minimum indentation
            min_indent = float('inf')
            for line in code_lines:
                if line.strip():  # Skip empty lines
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            # Remove consistent indentation
            dedented_code = '\n'.join(line[min_indent:] if line.strip() else ''
                                      for line in code_lines)

            # Combine code with answer template
            combined_code = f"""
{dedented_code}
# Format the answer template
answer_text = f'''{segments['answer']}'''
"""

            exec(combined_code, namespace)
            results['answer'] = namespace.get('answer_text')

        if 'chart' in segments:
            # Properly dedent the chart code
            if "No" in segments['chart']:
                pass
            else:
                chart_lines = segments['chart'].strip().split('\n')
                chart_lines = [x for x in chart_lines if 'plt.show' not in x]
                # Find minimum indentation
                min_indent = float('inf')
                for line in chart_lines:
                    if line.strip():  # Skip empty lines
                        indent = len(line) - len(line.lstrip())
                        min_indent = min(min_indent, indent)
                # Remove consistent indentation
                dedented_chart = '\n'.join(line[min_indent:] if line.strip() else ''
                                           for line in chart_lines)

                plot_path = os.path.join(PLOT_DIR, f"plot_{uuid.uuid4().hex}.png")
                dedented_chart += f"\nplt.savefig('{plot_path}', bbox_inches='tight')"

                exec(dedented_chart, namespace)

                results['figure'] = plot_path

        return results

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        return results
