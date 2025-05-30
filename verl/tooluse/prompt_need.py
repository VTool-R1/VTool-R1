import json
import random
import cv2
import numpy as np
random.seed(0)


MULTIMODAL_ASSISTANT_MESSAGE = """You are a helpful multimodal AI assistant.
Solve tasks using your vision, coding, and language skills.
The task can be free-form or multiple-choice questions.
You can answer the user's question about images. If you are not sure, you can coding 
You are coding in a Python jupyter notebook environment.
You can suggest python code (in a python coding block) for the user to execute. In a dialogue, all your codes are executed with the same jupyter kernel, so you can use the variables, working states.. in your earlier code blocks.
Solve the task step by step if you need to. 
The task may be a vision-language task and require several steps. You can write code to process images, text, or other data in a step. Give your code to the user to execute. The user may reply with the text and image outputs of the code execution. You can use the outputs to proceed to the next step, with reasoning, planning, or further coding.
If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
All images should be stored in PIL Image objects. The notebook has imported 'Image' from 'PIL' package and 'display' from 'IPython.display' package. If you want to read the image outputs of your code, use 'display' function to show the image in the notebook. The user will send the image outputs to you.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.

For each turn, you should first do a "THOUGHT", based on the images and text you see.
If you think you get the answer to the intial user request, you can reply with "ANSWER: <your answer>" and ends with "TERMINATE".
"""


class ChartQAPrompt_mix_orig:
    
    def __init__(self) -> None:
        return
    
    def initial_prompt(self, ex, n_image) -> str:
        initial_prompt = """Here are some tools that can help you. All are python codes. They are in tools.py and will be imported for you.
You will be given a chart figure: image_1 and a question.
Notice that you, as an AI assistant, are not good at answering questions when there are too many unnecessary and irrelevant information. 
If you are dealing with a vertical bar chart figure, you should determine which are the relevant x values to the question, and specify them in a python list. You should use the given x value names. 
If you are dealing with a horizontal bar chart figure, you should also determine which are the relevant y values to the question, and specify them in a python list. You should use the given y value names.
Below are the tools in tools.py:
```python
def focus_on_x_values_with_mask(image, x_values_to_focus_on, all_x_values_bounding_boxes):
    \"\"\"
    This function is useful when you want to focus on some specific x values in the image.
    It does this by masking out the x values that are not needed.
    This function is especially useful for vertical bar charts.
    For example, you can focus on the x values in a chart that are relevant to your analysis and ignore the rest.
    Return the masked image.

    Args:
        image (PIL.Image.Image): the input image
        x_values_to_focus_on (List[str]): a list of x values to focus on. 
        all_x_values_bounding_boxes (Dict[Dict]): a dictionary of bounding boxes for all x values in the image. key is x value and value is the bounding box of that x value. Each bounding box is in the format {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}.
    
    Returns:
        image_with_focused_x_values (PIL.Image.Image): the image with specified x values focused on

    Example:
        image = Image.open("sample_img.jpg")
        image_with_focused_x_values = focus_on_x_values(image, ["2005", "2006"], {"2005": {'x1': 0.1, 'y1': 0.1, 'x2': 0.3, 'y2': 0.9}, "2006": {'x1': 0.4, 'y1': 0.1, 'x2': 0.6, 'y2': 0.9}, "2007": {'x1': 0.7, 'y1': 0.1, 'x2': 0.9, 'y2': 0.9}})
        display(image_with_focused_x_values)
    \"\"\"

def focus_on_y_values_with_mask(image, y_values_to_focus_on, all_y_values_bounding_boxes):
    \"\"\"
    This function is useful when you want to focus on some specific y values in the image.
    It does this by masking out the y values that are not needed.
    This function is especially useful for horizontal bar charts.
    For example, you can focus on the y values in a chart that are relevant to your analysis and ignore the rest.
    Return the masked image.

    Args:
        image (PIL.Image.Image): the input image
        y_values_to_focus_on (List[str]): a list of y values to focus on. 
        all_y_values_bounding_boxes (Dict[Dict]): a dictionary of bounding boxes for all y values in the image. key is y value and value is the bounding box of that y value. Each bounding box is in the format {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}.

    Returns:
        image_with_focused_y_values (PIL.Image.Image): the image with specified y values focused on

    Example:
        image = Image.open("sample_img.jpg")
        image_with_focused_y_values = focus_on_y_values(image, ["0", "10"], {"0": {'x1': 0.1, 'y1': 0.1, 'x2': 0.9, 'y2': 0.15}, "10": {'x1': 0.1, 'y1': 0.2, 'x2': 0.9, 'y2': 0.5}, "20": {'x1': 0.1, 'y1': 0.6, 'x2': 0.9, 'y2': 0.9}})
    \"\"\"

def focus_on_x_values_with_draw(image, x_values_to_focus_on, all_x_values_bounding_boxes):
    \"\"\"
    This function is useful when you want to focus on some specific x values in the image.
    It does this by drawing a red box around the x values that need to be focused on.
    This function is especially useful for vertical bar charts.
    For example, you can focus on the x values in a chart that are relevant to your analysis.
    Return the masked image.

    Args:
        image (PIL.Image.Image): the input image
        x_values_to_focus_on (List[str]): a list of x values to focus on. 
        all_x_values_bounding_boxes (Dict[Dict]): a dictionary of bounding boxes for all x values in the image. key is x value and value is the bounding box of that x value. Each bounding box is in the format {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}.
    
    Returns:
        image_with_focused_x_values (PIL.Image.Image): the image with specified x values focused on

    Example:
        image = Image.open("sample_img.jpg")
        image_with_focused_x_values = focus_on_x_values(image, ["2005", "2006"], {"2005": {'x1': 0.1, 'y1': 0.1, 'x2': 0.3, 'y2': 0.9}, "2006": {'x1': 0.4, 'y1': 0.1, 'x2': 0.6, 'y2': 0.9}, "2007": {'x1': 0.7, 'y1': 0.1, 'x2': 0.9, 'y2': 0.9}})
        display(image_with_focused_x_values)
    \"\"\"

def focus_on_y_values_with_draw(image, y_values_to_focus_on, all_y_values_bounding_boxes):
    \"\"\"
    This function is useful when you want to focus on some specific y values in the image.
    It does this by drawing a red box around the y values that need to be focused on.
    This function is especially useful for horizontal bar charts.
    For example, you can focus on the y values in a chart that are relevant to your analysis.
    Return the masked image.

    Args:
        image (PIL.Image.Image): the input image
        y_values_to_focus_on (List[str]): a list of y values to focus on. 
        all_y_values_bounding_boxes (Dict[Dict]): a dictionary of bounding boxes for all y values in the image. key is y value and value is the bounding box of that y value. Each bounding box is in the format {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}.

    Returns:
        image_with_focused_y_values (PIL.Image.Image): the image with specified y values focused on

    Example:
        image = Image.open("sample_img.jpg")
        image_with_focused_y_values = focus_on_y_values(image, ["0", "10"], {"0": {'x1': 0.1, 'y1': 0.1, 'x2': 0.9, 'y2': 0.15}, "10": {'x1': 0.1, 'y1': 0.2, 'x2': 0.9, 'y2': 0.5}, "20": {'x1': 0.1, 'y1': 0.6, 'x2': 0.9, 'y2': 0.9}})
    \"\"\"

def focus_on_x_values_with_highlight(image, x_values_to_focus_on, all_x_values_bounding_boxes):
    \"\"\"
    This function is useful when you want to focus on some specific x values in the image.
    It does this by adding light transparent red highlight to the x values that need to be focused on.
    This function is especially useful for vertical bar charts.
    For example, you can focus on the x values in a chart that are relevant to your analysis.
    Return the masked image.

    Args:
        image (PIL.Image.Image): the input image
        x_values_to_focus_on (List[str]): a list of x values to focus on. 
        all_x_values_bounding_boxes (Dict[Dict]): a dictionary of bounding boxes for all x values in the image. key is x value and value is the bounding box of that x value. Each bounding box is in the format {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}.
    
    Returns:
        image_with_focused_x_values (PIL.Image.Image): the image with specified x values focused on

    Example:
        image = Image.open("sample_img.jpg")
        image_with_focused_x_values = focus_on_x_values(image, ["2005", "2006"], {"2005": {'x1': 0.1, 'y1': 0.1, 'x2': 0.3, 'y2': 0.9}, "2006": {'x1': 0.4, 'y1': 0.1, 'x2': 0.6, 'y2': 0.9}, "2007": {'x1': 0.7, 'y1': 0.1, 'x2': 0.9, 'y2': 0.9}})
        display(image_with_focused_x_values)
    \"\"\"

def focus_on_y_values_with_highlight(image, y_values_to_focus_on, all_y_values_bounding_boxes):
    \"\"\"
    This function is useful when you want to focus on some specific y values in the image.
    It does this by adding light transparent red highlight to the y values that need to be focused on.
    This function is especially useful for horizontal bar charts.
    For example, you can focus on the y values in a chart that are relevant to your analysis.
    Return the masked image.

    Args:
        image (PIL.Image.Image): the input image
        y_values_to_focus_on (List[str]): a list of y values to focus on. 
        all_y_values_bounding_boxes (Dict[Dict]): a dictionary of bounding boxes for all y values in the image. key is y value and value is the bounding box of that y value. Each bounding box is in the format {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}.

    Returns:
        image_with_focused_y_values (PIL.Image.Image): the image with specified y values focused on

    Example:
        image = Image.open("sample_img.jpg")
        image_with_focused_y_values = focus_on_y_values(image, ["0", "10"], {"0": {'x1': 0.1, 'y1': 0.1, 'x2': 0.9, 'y2': 0.15}, "10": {'x1': 0.1, 'y1': 0.2, 'x2': 0.9, 'y2': 0.5}, "20": {'x1': 0.1, 'y1': 0.6, 'x2': 0.9, 'y2': 0.9}})
    \"\"\"
```
# GOAL #: Based on the above tools, I want you to reason about how to solve the # USER REQUEST # and generate the actions step by step (each action is a python jupyter notebook code block) to solve the request.
You may need to use the tools above to process the images and make decisions based on the visual outputs of the previous code blocks.
Your visual ability is not perfect, so you should use these tools to assist you in reasoning about the images.
The jupyter notebook has already executed the following code to import the necessary packages:
```python
from PIL import Image
from IPython.display import display
from tools import *
```

# REQUIREMENTS #:
1. The generated actions can resolve the given user request # USER REQUEST # perfectly. The user request is reasonable and can be solved. Try your best to solve the request.
2. The arguments of a tool must be the same format specified in # TOOL LIST #;
3. If you think you got the answer, use ANSWER: <your answer> Please extract the final answer in FINAL ANSWER: <final answer> and ends with TERMINATE.
4. All images in the initial user request are stored in PIL Image objects named image_1, image_2, ..., image_n. You can use these images in your code blocks. Use display() function to show the image in the notebook for you too see.
5. Use as few tools as possible. Only use the tools for the use cases written in the tool description. You can use multiple tools in a single action.
6. If you do not think you have enough information to answer the question on the images returned by the tools, you should directly answer the question based on the original image.
7. If all the x values are relevant to the question for a vertical bar chart, you do not need to focus on any specific x values. You should directly answer the question based on the original image.
8. If all the y values are relevant to the question for a horizontal bar chart, you do not need to focus on any specific y values. You should directly answer the question based on the original image.
9. In most situations, you only need to draw or highlight. Only use mask tools when necessary.
Below are some examples of how to use the tools to solve the user requests. You can refer to them for help. You can also refer to the tool descriptions for more information.

# EXAMPLE: Simple question that does not require any tool
# USER REQUEST #: <A image here> What is the title of this chart?
# USER Bounding Box Info: x_values_bbox, storing x values and coordinates. y_values_bbox, storing x values and coordinates. The x values in the image are: ["2005", "2006", "2007"]. The y values in the image are: [].
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: The question does not require any tool. I can see the title of the table is "Customer Information".
ACTION 0: No action needed.
ANSWER: The title of the table is "Customer Information". FINAL ANSWER: Customer Information. TERMINATE

# EXAMPLE:  Focus on specific x values in vertical bar chart image
# USER REQUEST #: <A image here> What's the annual gain in China?
# USER Bounding Box Info: x_values_bbox, storing x values and coordinates. y_values_bbox, storing x values and coordinates. The x values in the image are: ["China", "USA", "UK"]. The y values in the image are: [].
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: This is a vertical bar chart image, and I need to focus on the part when x axis value equals 'China' and find out the annual gain.
ACTION 0:
```python
image_with_focused_x_values = focus_on_x_values_with_highlight(image_1, ["China"], x_values_bbox)
display(image_with_focused_x_values)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
If you can get the answer, please reply with ANSWER: <your answer> Please extract the final answer in FINAL ANSWER: <final answer> and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: I can see that the annual gain in China is $100,000.
ACTION 1: No action needed.
ANSWER: The annual gain in China is $100,000. FINAL ANSWER: 100000. TERMINATE

# EXAMPLE:  Focus on specific y values in the horizontal bar chart image
# USER REQUEST #: <A image here> How many games did Josh win? 
# USER Bounding Box Info: x_values_bbox, storing x values and coordinates. y_values_bbox, storing x values and coordinates. The x values in the image are: []. The y values in the image are: ["Josh", "Alice", "Bob"]. 
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: This is a horizontal bar chart image, I need to focus on the part when y axis value equals 'Josh' and find out how many games he won.
ACTION 0:
```python
image_with_focused_y_values = focus_on_y_values_with_draw(image_1, ["Josh"], y_values_bbox)
display(image_with_focused_y_values)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
If you can get the answer, please reply with ANSWER: <your answer> Please extract the final answer in FINAL ANSWER: <final answer> and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: I can see that Josh won 3 games.
ACTION 1: No action needed.
ANSWER: Josh won 3 games. FINAL ANSWER: 3. TERMINATE

# Example: Should focus on every x and y in the image so should not use any tool
# USER REQUEST #: <A image here> What is the average spending by Dan over the years in the figure?
# USER Bounding Box Info: x_values_bbox, storing x values and coordinates. y_values_bbox, storing x values and coordinates. The x values in the image are: ['Year', 'Spending']. The y values in the image are: ['2001', '2002', '2003'].
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: The question does not require any tool. I can see that the spending over the years by Dan are 12, 15, 18. The average spending by Dan over the years is (12 + 15 + 18) / 3 = 15.
ACTION 0: No action needed.
ANSWER: The average spending by Dan over the years is 15. FINAL ANSWER: 15. TERMINATE

"""
        self.ex = ex
        prompt = initial_prompt
        self.question = ex["query"]
        image_path_code = ex["figure_path"]
        self.original_image_path = image_path_code
        x_values = ex['x_values']
        y_values = ex['y_values']

        prompt += f"""# USER REQUEST #: <img src='{image_path_code}'> {self.question}
# USER Bounding Box Info: x_values_bbox, storing x values and coordinates. y_values_bbox, storing x values and coordinates. The x values in the image are: {x_values}. The y values in the image are: {y_values}.
# USER IMAGE stored in image_1, as PIL image. 
Now please generate only THOUGHT 0 and ACTION 0 in RESULT. If no action needed, also reply with ANSWER: <your answer> Please extract the final answer in FINAL ANSWER: <final answer> and ends with TERMINATE:\n# RESULT #:\n"""
        return prompt
    
    def get_parsing_feedback(self, error_message: str, error_code: str) -> str:
        return f"OBSERVATION: Parsing error. Error code: {error_code}, Error message:\n{error_message}\nPlease fix the error and generate the fixed code, in the next THOUGHT and ACTION."
    
    def get_exec_feedback(self, exit_code: int, output: str, image_path: list) -> str:    
        # if execution fails
        if exit_code != 0:
           return f"OBSERVATION: Execution error. Exit code: {exit_code}, Output:\n{output}\nPlease fix the error and generate the fixed code, in the next THOUGHT and ACTION."
        else:
            print(f'Edited image path: {image_path}')
            if len(image_path) == 0:
                image_path = [self.original_image_path]
            else:
                image_path.append(self.original_image_path)
            prompt = f"OBSERVATION: Execution success. The output is as follows:\n<img src='{image_path[0]}'>\nAnd a kind reminder that the original image is:\n<img src='{image_path[1]}'>\n"
            prompt += f"Answer the question {self.question}. You can turn the chart image into text and answer with step of thinking. \nReply with ANSWER: <your answer> Please extract the final answer in FINAL ANSWER: <final answer> and ends with TERMINATE."
            return prompt

def python_codes_for_images_reading(image_paths):
    code = ""
    for idx, path in enumerate(image_paths):
        code += f"""image_{idx+1} = Image.open("{path}").convert("RGB")\n"""
    return code

def python_codes_for_chart_bbox_load(image_paths, bbox_file):
    code = ""
    for idx, path in enumerate(image_paths):
        code += f"""image_{idx+1} = Image.open("{path}").convert("RGB")\n"""
    # bbox file is a json file, content being {"columns": columns_bbox, "rows": rows_bbox}
    code += f"""x_values_bbox = json.load(open('{bbox_file}', "r"))['x_values_bbox']
y_values_bbox = json.load(open('{bbox_file}', "r"))['y_values_bbox']
"""
    return code

def python_codes_for_table_bbox_load(image_paths, bbox_file):
    code = ""
    for idx, path in enumerate(image_paths):
        code += f"""image_{idx+1} = Image.open("{path}").convert("RGB")\n"""
    # bbox file is a json file, content being {"columns": columns_bbox, "rows": rows_bbox}
    code += f"""columns_bbox = json.load(open('{bbox_file}', "r"))['columns']
rows_bbox = json.load(open('{bbox_file}', "r"))['rows']
"""
    return code


class TablePrompt_Baseline:
    
    def __init__(self) -> None:
        return
    
    def initial_prompt(self, ex, n_image) -> str:
        initial_prompt = """"""
        prompt = initial_prompt

        # geometry 
        question = ex["prompt"]
        image_path_code = ex["figure_path"]
        prompt += f"Given the table figure <img src='{image_path_code}'>\n" + \
                 f"Solve the following question: {question}" + \
                 f"Please extract the final answer in FINAL ANSWER: <final answer> and ends with TERMINATE\n"
        return prompt
    
    def get_parsing_feedback(self, error_message: str, error_code: str) -> str:
        return f"OBSERVATION: Parsing error. Error code: {error_code}, Error message:\n{error_message}\nPlease fix the error and generate the fixed code, in the next THOUGHT and ACTION."
    
    def get_exec_feedback(self, exit_code: int, output: str, image_path: list) -> str:
        # if execution fails
        if exit_code != 0:
           return f"OBSERVATION: Execution error. Exit code: {exit_code}, Output:\n{output}\nPlease fix the error and generate the fixed code, in the next THOUGHT and ACTION."
        else:
            # bp()
            if len(image_path) != 0:
                prompt = f"OBSERVATION: Execution success. The output is as follows:\nOutput: {output}\nImage: <img src='{image_path[0]}'>\n"
            else:
                prompt = f"OBSERVATION: Execution success. The output is as follows:\nOutput: {output}\n"
            # prompt = f"OBSERVATION: Execution success. The output is as follows:\n<img src='{image_path[0]}'>\n"
            prompt += "If you can get the answer, please also reply with ANSWER: <your answer>, extract the final answer in FINAL ANSWER: <final answer> and ends with TERMINATE in the RESULT"
            return prompt
        

class TablePrompt_mix_col_row_cot:
    
    def __init__(self) -> None:
        return
    
    def initial_prompt(self, ex, n_image) -> str:
        initial_prompt = """Here are some tools that can help you. All are python codes. They are in tools.py and will be imported for you.
You will be given a table figure: image_1 and a question.
Notice that you, as an AI assistant, are not good at answering questions when there are too many unnecessary and irrelevant information. You should determine which are the relevant columns to the question, and specify them in a python list. You should use the given column headers.
You should also determine which are the relevant rows to the question, and specify them in a python list. You should use the given row headers.
You could select the tools to focus on some columns / rows, or mask out some columns / rows. Use whichever tool you think is more appropriate.
Below are the tools in tools.py:
```python
def focus_on_columns_with_highlight(image, columns_to_focus_on, all_columns_bounding_boxes):
    \"\"\"
    This function is useful when you want to focus on some specific columns of the image.
    It does this by adding light transparent red highlight to the columns that need to be focused on.
    For example, you can focus on the columns in a table that are relevant to your analysis.
    Return the drawed image.

    Args:
        image (PIL.Image.Image): the input image
        columns_to_mask (List[str]): a list of column names to focus on.
        all_columns_bounding_boxes (Dict[Dict]]): a dictionary of bounding boxes for all columns in the image. key is column name and value is the bounding box of that column. Each bounding box is in the format {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}.

    Returns:
        image_with_focused_columns (PIL.Image.Image): the image with specified columns focused on
        
    Example:
        image = Image.open("sample_img.jpg")
        image_with_focused_columns = focus_on_columns_with_highlight(image, ["Year", "Name"], {"Year": {'x1': 0.1, 'y1': 0.1, 'x2': 0.3, 'y2': 0.9}, "Team": {'x1': 0.4, 'y1': 0.1, 'x2': 0.6, 'y2': 0.9}, "Name": {'x1': 0.7, 'y1': 0.1, 'x2': 0.9, 'y2': 0.9}})
        display(image_with_focused_columns)
    \"\"\"

def focus_on_rows_with_highlight(image, rows_to_focus_on, all_rows_bounding_boxes):
    \"\"\"
    This function is useful when you want to focus on some specific rows of the image.
    It does this by adding light transparent red highlight to the rows that need to be focused on.
    For example, you can focus on the rows in a table that are relevant to your analysis.
    Return the drawed image.
    
    Args:
        image (PIL.Image.Image): the input image
        rows_to_focus_on (List[str]): a list of row headers to focus on.
        all_rows_bounding_boxes (Dict[Dict]): a dictionary of bounding boxes for all rows in the image. key is row header and value is the bounding box of that row. Each bounding box is in the format {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}.
    
    Returns:
        image_with_focused_rows (PIL.Image.Image): the image with specified rows focused on

    Example:
        image = Image.open("sample_img.jpg")
        image_with_focused_rows = focus_on_rows_with_highlight(image, ["1972"], ["Year": {'x1': 0.1, 'y1': 0.1, 'x2': 0.9, 'y2': 0.15}, "1969": {'x1': 0.1, 'y1': 0.2, 'x2': 0.9, 'y2': 0.5}, "1972": {'x1': 0.1, 'y1': 0.6, 'x2': 0.9, 'y2': 0.9}])
        display(image_with_focused_rows)
    \"\"\"

def focus_on_columns_with_mask(image, columns_to_focus_on, all_columns_bounding_boxes):
    \"\"\"
    This function is useful when you want to focus on some specific columns of the image.
    It does this by masking out the columns that are not needed.
    For example, you can focus on the columns in a table that are relevant to your analysis and ignore the rest.
    Return the masked image.

    Args:
        image (PIL.Image.Image): the input image
        columns_to_mask (List[str]): a list of column names to focus on.
        all_columns_bounding_boxes (Dict[Dict]]): a dictionary of bounding boxes for all columns in the image. key is column name and value is the bounding box of that column. Each bounding box is in the format {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}.

    Returns:
        image_with_focused_columns (PIL.Image.Image): the image with specified columns focused on
        
    Example:
        image = Image.open("sample_img.jpg")
        image_with_focused_columns = focus_on_columns(image, ["Year", "Name"], {"Year": {'x1': 0.1, 'y1': 0.1, 'x2': 0.3, 'y2': 0.9}, "Team": {'x1': 0.4, 'y1': 0.1, 'x2': 0.6, 'y2': 0.9}, "Name": {'x1': 0.7, 'y1': 0.1, 'x2': 0.9, 'y2': 0.9}})
        display(image_with_focused_columns)
    \"\"\"

def focus_on_rows_with_mask(image, rows_to_focus_on, all_rows_bounding_boxes):
    \"\"\"
    This function is useful when you want to focus on some specific rows of the image.
    It does this by masking out the rows that are not needed.
    For example, you can focus on the rows in a table that are relevant to your analysis and ignore the rest.
    Return the masked image.
    
    Args:
        image (PIL.Image.Image): the input image
        rows_to_focus_on (List[str]): a list of row headers to focus on.
        all_rows_bounding_boxes (Dict[Dict]): a dictionary of bounding boxes for all rows in the image. key is row header and value is the bounding box of that row. Each bounding box is in the format {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}.
    
    Returns:
        image_with_focused_rows (PIL.Image.Image): the image with specified rows focused on

    Example:
        image = Image.open("sample_img.jpg")
        image_with_focused_rows = focus_on_rows(image, ["1972"], ["Year": {'x1': 0.1, 'y1': 0.1, 'x2': 0.9, 'y2': 0.15}, "1969": {'x1': 0.1, 'y1': 0.2, 'x2': 0.9, 'y2': 0.5}, "1972": {'x1': 0.1, 'y1': 0.6, 'x2': 0.9, 'y2': 0.9}])
        display(image_with_focused_rows)
    \"\"\"
    
def focus_on_columns_with_draw(image, columns_to_focus_on, all_columns_bounding_boxes):
    \"\"\"
    This function is useful when you want to focus on some specific columns of the image.
    It does this by drawing a red box around the columns that need to be focused on.
    For example, you can focus on the columns in a table that are relevant to your analysis.
    Return the drawed image.

    Args:
        image (PIL.Image.Image): the input image
        columns_to_mask (List[str]): a list of column names to focus on.
        all_columns_bounding_boxes (Dict[Dict]]): a dictionary of bounding boxes for all columns in the image. key is column name and value is the bounding box of that column. Each bounding box is in the format {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}.

    Returns:
        image_with_focused_columns (PIL.Image.Image): the image with specified columns focused on
        
    Example:
        image = Image.open("sample_img.jpg")
        image_with_focused_columns = focus_on_columns(image, ["Year", "Name"], {"Year": {'x1': 0.1, 'y1': 0.1, 'x2': 0.3, 'y2': 0.9}, "Team": {'x1': 0.4, 'y1': 0.1, 'x2': 0.6, 'y2': 0.9}, "Name": {'x1': 0.7, 'y1': 0.1, 'x2': 0.9, 'y2': 0.9}})
        display(image_with_focused_columns)
    \"\"\"

def focus_on_rows_with_draw(image, rows_to_focus_on, all_rows_bounding_boxes):
    \"\"\"
    This function is useful when you want to focus on some specific rows of the image.
    It does this by drawing a red box around the rows that need to be focused on.
    For example, you can focus on the rows in a table that are relevant to your analysis.
    Return the drawed image.
    
    Args:
        image (PIL.Image.Image): the input image
        rows_to_focus_on (List[str]): a list of row headers to focus on.
        all_rows_bounding_boxes (Dict[Dict]): a dictionary of bounding boxes for all rows in the image. key is row header and value is the bounding box of that row. Each bounding box is in the format {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}.
    
    Returns:
        image_with_focused_rows (PIL.Image.Image): the image with specified rows focused on

    Example:
        image = Image.open("sample_img.jpg")
        image_with_focused_rows = focus_on_columns_with_highlight(image, ["1972"], ["Year": {'x1': 0.1, 'y1': 0.1, 'x2': 0.9, 'y2': 0.15}, "1969": {'x1': 0.1, 'y1': 0.2, 'x2': 0.9, 'y2': 0.5}, "1972": {'x1': 0.1, 'y1': 0.6, 'x2': 0.9, 'y2': 0.9}])
        display(image_with_focused_rows)
    \"\"\"
```
# GOAL #: Based on the above tools, I want you to reason about how to solve the # USER REQUEST # and generate the actions step by step (each action is a python jupyter notebook code block) to solve the request.
You may need to use the tools above to process the images and make decisions based on the visual outputs of the previous code blocks.
Your visual ability is not perfect, so you should use these tools to assist you in reasoning about the images.
The jupyter notebook has already executed the following code to import the necessary packages:
```python
from PIL import Image
from IPython.display import display
from tools import *
```

# REQUIREMENTS #:
1. The generated actions can resolve the given user request # USER REQUEST # perfectly. The user request is reasonable and can be solved. Try your best to solve the request.
2. The arguments of a tool must be the same format specified in # TOOL LIST #;
3. If you think you got the answer, use ANSWER: <your answer> Please extract the final answer in FINAL ANSWER: <final answer> and ends with TERMINATE.
4. All images in the initial user request are stored in PIL Image objects named image_1, image_2, ..., image_n. You can use these images in your code blocks. Use display() function to show the image in the notebook for you too see.
5. Use as few tools as possible. Only use the tools for the use cases written in the tool description. You can use multiple tools in a single action.
6. If you have multiple answers, please separate them with || marks. For example, if the answer is 'Alice' and 'Bob', you should write 'Alice||Bob'.
7. When you focus on columns in the image, most like you need to look at multiple columns instead of a single one. 
8. If you do not think you have enough information to answer the question on the images returned by the tools, you should directly answer the question based on the original image.
Below are some examples of how to use the tools to solve the user requests. You can refer to them for help. You can also refer to the tool descriptions for more information.

# EXAMPLE: Simple question that does not require any tool
# USER REQUEST #: <A image here> What is the title of this table?
# USER Bounding Box Info: columns_bbox, where keys are column headers and values are column bounding boxes. rows_bbox, where keys are row headers and values are row bounding boxes. The columns in the image are: ["Grade", "Mentor", "Salary"]. The rows in the image start with: ["Grade", "A", "B", "C"].
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: The question does not require any tool. I can see the title of the table is "Customer Information".
ACTION 0: No action needed.
ANSWER: The title of the table is "Customer Information". FINAL ANSWER: Customer Information. TERMINATE

# EXAMPLE:  Focus on specific columns in the image
# USER REQUEST #: <A image here> Who had the same game version as John Roth?
# USER Bounding Box Info: columns_bbox, where keys are column headers and values are column bounding boxes. rows_bbox, where keys are row headers and values are row bounding boxes. The columns in the image are: ['Manager Name', 'Game Version', 'Game Score']. The rows in the image start with: ['Manager Name', 'John Roth', 'Alice Smith', 'Bob Johnson'].
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: To identify who had the same game version as John Roth, I need to focus on the 'Game Version' column and the 'Manager Name' column. Also, I need to focus on all the rows so I do not need to focus on some specific rows.
ACTION 0:
```python
image_with_focused_columns = focus_on_columns_with_draw(image_1, ["Game Version", "Manager Name"], columns_bbox)
display(image_with_focused_columns)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
If you can get the answer, please reply with ANSWER: <your answer> Please extract the final answer in FINAL ANSWER: <final answer> and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: Now I can see the 'Game Version' column and the 'Manager Name' column more clearly. The game version of John Roth is 'v1.2'. Other people with the same game version are 'Alice Smith' and 'Bob Johnson'.
ACTION 1: No action needed.
ANSWER: 'Alice Smith' and 'Bob Johnson' had the same game version as John Roth are. FINAL ANSWER: Alice Smith||Bob Johnson. TERMINATE

# EXAMPLE:  Focus on specific rows in the image
# USER REQUEST #: <A image here> How many games did Josh win after 1996? 
# USER Bounding Box Info: columns_bbox, where keys are column names and values are column bounding boxes. rows_bbox, where keys are row headers and values are row bounding boxes. The columns in the image are: ["Rank", "Year", "Score", "Month"]. The rows in the image start with: ["Rank", "0", "1", "2", "3", "4", "5"].
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: This table is about the games played by Josh that he won. I do not need to focus on any specific columns. I need to focus on the rows with the year after 1996. Three rows have year after 1996: one has year 1997 and this row starts with "3", one has year 1998 and this row starts with "4", and one has year 1999 and this row starts with "5". So I will focus on the rows with row starters "3", "4", and "5".
ACTION 0:
```python
image_with_focused_rows = focus_on_rows_with_highlight(image_1, ["3", "4", "5"], rows_bbox)
display(image_with_focused_rows)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
If you can get the answer, please reply with ANSWER: <your answer> Please extract the final answer in FINAL ANSWER: <final answer> and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: I can see that Josh won 3 games after 1996.
ACTION 1: No action needed.
ANSWER: Josh won 3 games after 1996. FINAL ANSWER: 3. TERMINATE

# EXAMPLE:  Focus on specific columns and specific rows in the image
# USER REQUEST #: <A image here> what is the sum of annual earnings after 2006? 
# USER Bounding Box Info: columns_bbox, where keys are column names and values are column bounding boxes. rows_bbox, where keys are row headers and values are row bounding boxes. The columns in the image are: ["Index", "Year", "Cost", "Earning"]. The rows in the image start with: ["Index", "0", "1", "2", "3", "4", "5"].
# USER IMAGE stored in image_1, as PIL image.
# RESULT #:
THOUGHT 0: I need to focus on the 'Year' column and the 'Earning' column. I also need to focus on the rows with years after 2006. The row with year being 2006 starts with "3". So I will focus on the rows starting with "4", and "5".
ACTION 0:
```python
image_with_focused_columns = focus_on_columns_with_mask(image_1, ["Year", "Earning"], columns_bbox)
image_with_focused_rows = focus_on_rows_with_draw(image_with_focused_columns, ["4", "5"], rows_bbox)
display(image_with_focused_rows)
```
OBSERVATION: Execution success. The output is as follows:
<the image outputs of the previous code is here.>
If you can get the answer, please reply with ANSWER: <your answer> Please extract the final answer in FINAL ANSWER: <final answer> and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.
THOUGHT 1: I can see that the annual earnings after 2006 are $165,498 and $198,765. The sum of the annual earnings after 2006 is $364,263.
ACTION 1: No action needed.
ANSWER: The sum of the annual earnings after 2006 is $364,263. FINAL ANSWER: 364263. TERMINATE

"""
        self.ex = ex
        self.question = ex["query"]
        column_headers = ex["column_headers"]
        row_headers = ex["row_starters"]
        image_path_code = ex["figure_path"]
        self.original_image_path = image_path_code
        prompt = initial_prompt
        prompt += f"""# USER REQUEST #: <img src='{image_path_code}'> {self.question}
# USER Bounding Box Info: columns_bbox, where keys are column headers and values are column bounding boxes. rows_bbox, where keys row headers and values are row bounding boxes. The columns in the image are: {column_headers}. The rows in the image start with: {row_headers}.
# USER IMAGE stored in image_1, as PIL image. 
Now please generate only THOUGHT 0 and ACTION 0 in RESULT. If no action needed, also reply with ANSWER: <your answer> Please extract the final answer in FINAL ANSWER: <final answer> and ends with TERMINATE:\n# RESULT #:\n"""
        return prompt
    
    def get_parsing_feedback(self, error_message: str, error_code: str) -> str:
        return f"OBSERVATION: Parsing error. Error code: {error_code}, Error message:\n{error_message}\nPlease fix the error and generate the fixed code, in the next THOUGHT and ACTION."
    
    def get_exec_feedback(self, exit_code: int, output: str, image_path: list) -> str:    
        # if execution fails
        if exit_code != 0:
           return f"OBSERVATION: Execution error. Exit code: {exit_code}, Output:\n{output}\nPlease fix the error and generate the fixed code, in the next THOUGHT and ACTION."
        else:
            print(f'Edited image path: {image_path}')
            if len(image_path) == 0:
                image_path = [self.original_image_path]
            prompt = f"OBSERVATION: Execution success. The output is as follows:\n<img src='{image_path[0]}'>\n"
            prompt += f"Answer the question {self.question}. You can turn the table image into text and answer with step of thinking. \nReply with ANSWER: <your answer> Please extract the final answer in FINAL ANSWER: <final answer> and ends with TERMINATE."
            return prompt


class CharXivPrompt_mask_cv2:    
    def __init__(self) -> None:
        return
    
    def initial_prompt(self, ex, n_image) -> str:
        self.ex = ex
        question = ex["prompt"]
        self.question = question
        image_path_code = ex["figure_path"]
        self.original_image_path = image_path_code
        # Find subfigure coordinates with cv2
        image = cv2.imread(image_path_code)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        gaussian = cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)
        edged = cv2.Canny(gaussian, 100, 200) 
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours = sorted(contours, key= len, reverse=True)[:10]
        longest_contour = sorted_contours[0]
        x,y,w,h = cv2.boundingRect(np.concatenate(longest_contour))
        longest_box = w + h

        sorted_bbox = []
        for c in sorted_contours:
            x,y,w,h = cv2.boundingRect(np.concatenate(c))
            if longest_box - 10 < w + h <= longest_box + 10:
                repeat = False
                for bb in sorted_bbox:
                    if abs(bb['x1'] - x) < 10 and abs(bb['y1'] - y) < 10 and abs(bb['x2'] - (x+w)) < 10 and abs(bb['y2'] - (y+h)) < 10:
                        repeat = True
                if not repeat:
                    sorted_bbox.append({'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h})
        sorted_bbox_json = json.dumps(sorted_bbox)

        prompt = f"You are given the chart figure <img src='{image_path_code}'> with image path being {image_path_code}, and need to solve the following question: {question}"
        prompt += f"""
        If the image includes multiple subplots, and the question only requires looking at some of the subplots, not all of them: find the subplots that do not need to look to and mask them out with white color, operating on the original image. If the question needs to look at all subplots, do not mask out anything and simply return the original image.
        The steps are this: find out the subplots to focus on and subplots to ignore. Mask out the subplots to ignore with white color using python code.        
        You should use the bounding boxes candidates provided and directly edit upon the original image. Returned edited image in PIL image format. If the subplot you want to focus on is not in the bounding box candidates, do not edit anything and return the original image.
        Bounding box candidates: {sorted_bbox_json}\n"""
        return prompt
    
    def get_parsing_feedback(self, error_message: str, error_code: str) -> str:
        return f"OBSERVATION: Parsing error. Error code: {error_code}, Error message:\n{error_message}\nPlease fix the error and generate the fixed code, in the next THOUGHT and ACTION."
    
    def get_exec_feedback(self, exit_code: int, output: str, image_path: list) -> str:    
        if exit_code != 0:
           return f"OBSERVATION: Execution error. Exit code: {exit_code}, Output:\n{output}\nPlease fix the error and generate the fixed code, in the next THOUGHT and ACTION."
        else:
            print(f'Edited image path: {image_path}')
            if len(image_path) == 0:
                image_path = [self.original_image_path]
            prompt = f"OBSERVATION: Execution success. The image is as follows:\n<img src='{image_path[0]}'>\n"
            prompt += f"Answer the question {self.question} based on the image.  You can turn the chart image into text and answer with step of thinking. \nReply with ANSWER: <your answer> Please extract the final answer in FINAL ANSWER: <final answer> and ends with TERMINATE."
            return prompt
