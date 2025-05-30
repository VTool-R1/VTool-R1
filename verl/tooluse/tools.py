import cv2, json
import numpy as np
from PIL import Image, ImageDraw


# TableVQA Image Tools
def focus_on_columns_with_mask(image, columns_to_focus_on, all_columns_bounding_boxes):
    """
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
    """
    if len(all_columns_bounding_boxes) == 0 or len(columns_to_focus_on) == 0:
        return image
    # Create a drawing context for the image
    draw = ImageDraw.Draw(image, "RGBA")

    # Desipte the columns to focus on, mask out all other columns
    columns_to_mask = [column for column in all_columns_bounding_boxes.keys() if column not in columns_to_focus_on]
    #print(columns_to_mask)
    if len(columns_to_mask) == len(all_columns_bounding_boxes):
        # This means to mask all columns, must be mistake, so return the original image
        return image

    # Iterate over the columns to mask out
    for column_name in columns_to_mask:
        # Get the bounding box of the column
        column_bbox = all_columns_bounding_boxes[column_name]
        
        # Convert the bounding box to pixel coordinates
        # Define the region to mask out in the image as a tuple (x1, y1, x2, y2)
        x1 = int(column_bbox['x1'] + 2)
        y1 = int(column_bbox['y1'] + 2)
        x2 = int(column_bbox['x2'] - 2)
        y2 = int(column_bbox['y2'] - 2)
        draw.rectangle(((x1, y1), (x2, y2)), fill="white")

    image_with_focused_columns = image
    return image_with_focused_columns

def focus_on_rows_with_mask(image, rows_to_focus_on, all_rows_bounding_boxes):
    """
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
    """
    if len(rows_to_focus_on) == 0 or len(all_rows_bounding_boxes) == 0:
        return image

    # Create a drawing context for the image
    draw = ImageDraw.Draw(image, "RGBA")

    # Desipte the rows to focus on and the first row, mask out all other rows
    rows_to_mask = [row for row in list(all_rows_bounding_boxes.keys())[1:] if row not in rows_to_focus_on]
    #print(rows_to_mask)
    if len(rows_to_mask) == len(all_rows_bounding_boxes) - 1:
        # This means to mask all rows, must be mistake, so return the original image
        return image

    # Iterate over the rows to mask out
    for row_starter in rows_to_mask:
        if row_starter == list(all_rows_bounding_boxes.keys())[0]:
            # do not mask out the first row
            continue
        # Get the bounding box of the row
        row_bbox = all_rows_bounding_boxes[row_starter]
        
        # Convert the bounding box to pixel coordinates
        # Define the region to mask out in the image as a tuple (x1, y1, x2, y2)
        x1 = int(row_bbox['x1'] + 2)
        y1 = int(row_bbox['y1'] + 2)
        x2 = int(row_bbox['x2'] - 2)
        y2 = int(row_bbox['y2'] - 2)
        
        draw.rectangle(((x1, y1), (x2, y2)), fill="white")
    
    image_with_focused_rows = image
    return image_with_focused_rows

def focus_on_columns_with_draw(image, columns_to_focus_on, all_columns_bounding_boxes):
    """
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
    """
    if len(all_columns_bounding_boxes) == 0 or len(columns_to_focus_on) == 0:
        return image
    # Create a drawing context for the image
    draw = ImageDraw.Draw(image, "RGBA")

    # Draw a red box around the columns to focus on
    #print(columns_to_focus_on)
    # if len(columns_to_focus_on) == len(all_columns_bounding_boxes):
    #     # This means to draw all columns, no need, so return the original image
    #     return image

    # Iterate over the columns to mask out
    for column_name in columns_to_focus_on:
        # Get the bounding box of the column
        column_bbox = all_columns_bounding_boxes[column_name]
        
        # Convert the bounding box to pixel coordinates
        # Define the region to mask out in the image as a tuple (x1, y1, x2, y2)
        x1 = int(column_bbox['x1'] + 2)
        y1 = int(column_bbox['y1'] + 2)
        x2 = int(column_bbox['x2'] - 2)
        y2 = int(column_bbox['y2'] - 2)
        thickness = 5
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=thickness)

    image_with_focused_columns = image
    return image_with_focused_columns

def focus_on_rows_with_draw(image, rows_to_focus_on, all_rows_bounding_boxes):
    """
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
    """
    if len(rows_to_focus_on) == 0 or len(all_rows_bounding_boxes) == 0:
        return image

    # Create a drawing context for the image
    draw = ImageDraw.Draw(image, "RGBA")

    #print(rows_to_focus_on)
    # if len(rows_to_focus_on) >= len(all_rows_bounding_boxes) - 1:
    #     # This means to mask all rows, must be mistake, so return the original image
    #     return image

    # Iterate over the rows to mask out
    for row_starter in rows_to_focus_on:
        # Get the bounding box of the row
        row_bbox = all_rows_bounding_boxes[row_starter]
        
        # Convert the bounding box to pixel coordinates
        # Define the region to mask out in the image as a tuple (x1, y1, x2, y2)
        x1 = int(row_bbox['x1'] + 2)
        y1 = int(row_bbox['y1'] + 2)
        x2 = int(row_bbox['x2'] - 2)
        y2 = int(row_bbox['y2'] - 2)
        
        thickness = 5
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=thickness)
    
    image_with_focused_rows = image
    return image_with_focused_rows

def focus_on_columns_with_highlight(image, columns_to_focus_on, all_columns_bounding_boxes):
    """
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
        image_with_focused_columns = focus_on_columns(image, ["Year", "Name"], {"Year": {'x1': 0.1, 'y1': 0.1, 'x2': 0.3, 'y2': 0.9}, "Team": {'x1': 0.4, 'y1': 0.1, 'x2': 0.6, 'y2': 0.9}, "Name": {'x1': 0.7, 'y1': 0.1, 'x2': 0.9, 'y2': 0.9}})
        display(image_with_focused_columns)
    """
    if len(all_columns_bounding_boxes) == 0 or len(columns_to_focus_on) == 0:
        return image
    
    # Draw a highlight color on the columns to focus on
    mask = image.convert('RGBA').copy()
    mask_draw = ImageDraw.Draw(mask)
    #print(columns_to_focus_on)

    # Iterate over the columns to highlight
    for column_name in columns_to_focus_on:
        # Get the bounding box of the column
        column_bbox = all_columns_bounding_boxes[column_name]
        
        # Convert the bounding box to pixel coordinates
        # Define the region to mask out in the image as a tuple (x1, y1, x2, y2)
        x1 = int(column_bbox['x1'] + 2)
        y1 = int(column_bbox['y1'] + 2)
        x2 = int(column_bbox['x2'] - 2)
        y2 = int(column_bbox['y2'] - 2)
        mask_draw.rectangle(((x1, y1), (x2, y2)), fill=(255, 0, 0, 50))     
        
    # Composite the overlay with the mask over the original image
    image_with_focused_columns = Image.alpha_composite(image.convert('RGBA'), mask)
    return image_with_focused_columns

def focus_on_rows_with_highlight(image, rows_to_focus_on, all_rows_bounding_boxes):
    """
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
    """
    if len(rows_to_focus_on) == 0 or len(all_rows_bounding_boxes) == 0:
        return image
    
    # Draw a highlight color on the rows to focus on
    mask = image.convert('RGBA').copy()
    mask_draw = ImageDraw.Draw(mask)
    #print(rows_to_focus_on)

    # Iterate over the rows to mask out
    for row_starter in rows_to_focus_on:
        # Get the bounding box of the row
        row_bbox = all_rows_bounding_boxes[row_starter]
        
        # Convert the bounding box to pixel coordinates
        # Define the region to mask out in the image as a tuple (x1, y1, x2, y2)
        x1 = int(row_bbox['x1'] + 2)
        y1 = int(row_bbox['y1'] + 2)
        x2 = int(row_bbox['x2'] - 2)
        y2 = int(row_bbox['y2'] - 2)
        mask_draw.rectangle(((x1, y1), (x2, y2)), fill=(255, 0, 0, 50))     
        
    # Composite the overlay with the mask over the original image
    image_with_focused_rows = Image.alpha_composite(image.convert('RGBA'), mask)
    return image_with_focused_rows


# ChartQA Image Tools
def focus_on_x_values_with_mask(image, x_values_to_focus_on, all_x_values_bounding_boxes):
    """
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
    """
    if len(all_x_values_bounding_boxes) == 0 or len(x_values_to_focus_on) == 0:
        return image
    # Create a drawing context for the image
    draw = ImageDraw.Draw(image, "RGBA")

    # Desipte the x values to focus on, mask out all other x values
    x_values_to_mask = [x_value for x_value in all_x_values_bounding_boxes if x_value not in x_values_to_focus_on]
    #print(x_values_to_mask)
    if len(x_values_to_mask) == len(all_x_values_bounding_boxes):
        # This means to mask all x values, must be mistake, so return the original image
        return image
    
    # Iterate over the x values to mask out
    for x_value in x_values_to_mask:
        # Convert the bounding box to pixel coordinates
        # Define the region to mask out in the image as a tuple (x1, y1, x2, y2)
        x_value_bbox = all_x_values_bounding_boxes[x_value]
        x1 = int(x_value_bbox['x1'] + 2)
        y1 = int(x_value_bbox['y1'] + 2)
        x2 = int(x_value_bbox['x2'] - 2)
        y2 = int(x_value_bbox['y2'] - 2)
        draw.rectangle(((x1, y1), (x2, y2)), fill="white")
    
    image_with_focused_x_values = image
    return image_with_focused_x_values

def focus_on_y_values_with_mask(image, y_values_to_focus_on, all_y_values_bounding_boxes):
    """
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
    """
    if len(all_y_values_bounding_boxes) == 0 or len(y_values_to_focus_on) == 0:
        return image
    # Create a drawing context for the image
    draw = ImageDraw.Draw(image, "RGBA")

    # Desipte the y values to focus on, mask out all other y values
    y_values_to_mask = [y_value for y_value in all_y_values_bounding_boxes if y_value not in y_values_to_focus_on]
    #print(y_values_to_mask)
    if len(y_values_to_mask) == len(all_y_values_bounding_boxes):
        # This means to mask all y values, must be mistake, so return the original image
        return image
    
    # Iterate over the y values to mask out
    for y_value in y_values_to_mask:
        # Convert the bounding box to pixel coordinates
        # Define the region to mask out in the image as a tuple (x1, y1, x2, y2)
        y_value_bbox = all_y_values_bounding_boxes[y_value]
        x1 = int(y_value_bbox['x1'] + 2)
        y1 = int(y_value_bbox['y1'] + 2)
        x2 = int(y_value_bbox['x2'] - 2)
        y2 = int(y_value_bbox['y2'] - 2)
        draw.rectangle(((x1, y1), (x2, y2)), fill="white")

    image_with_focused_y_values = image
    return image_with_focused_y_values

def focus_on_x_values_with_draw(image, x_values_to_focus_on, all_x_values_bounding_boxes):
    """
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
    """
    if len(all_x_values_bounding_boxes) == 0 or len(x_values_to_focus_on) == 0:
        return image
    # Create a drawing context for the image
    draw = ImageDraw.Draw(image, "RGBA")

    #print(x_values_to_focus_on)
    
    # Iterate over the x values to draw
    for x_value in x_values_to_focus_on:
        # Convert the bounding box to pixel coordinates
        # Define the region to mask out in the image as a tuple (x1, y1, x2, y2)
        x_value_bbox = all_x_values_bounding_boxes[x_value]
        x1 = int(x_value_bbox['x1'] + 2)
        y1 = int(x_value_bbox['y1'] + 2)
        x2 = int(x_value_bbox['x2'] - 2)
        y2 = int(x_value_bbox['y2'] - 2)
        thickness = 5
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=thickness)
    
    image_with_focused_x_values = image
    return image_with_focused_x_values

def focus_on_y_values_with_draw(image, y_values_to_focus_on, all_y_values_bounding_boxes):
    """
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
    """
    if len(all_y_values_bounding_boxes) == 0 or len(y_values_to_focus_on) == 0:
        return image
    # Create a drawing context for the image
    draw = ImageDraw.Draw(image, "RGBA")

    #print(y_values_to_focus_on)

    # Iterate over the y values to draw
    for y_value in y_values_to_focus_on:
        # Convert the bounding box to pixel coordinates
        for ys in all_y_values_bounding_boxes:
            if y_value in ys or ys in y_value:
                y_value = ys
        try:
            y_value_bbox = all_y_values_bounding_boxes[y_value]
            x1 = int(y_value_bbox['x1'] + 2)
            y1 = int(y_value_bbox['y1'] + 2)
            x2 = int(y_value_bbox['x2'] - 2)
            y2 = int(y_value_bbox['y2'] - 2)
            thickness = 5
            draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=thickness)
        except Exception as e:
            print(f"Error: {e}")


    image_with_focused_y_values = image
    return image_with_focused_y_values

def focus_on_x_values_with_highlight(image, x_values_to_focus_on, all_x_values_bounding_boxes):
    """
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
    """
    if len(all_x_values_bounding_boxes) == 0 or len(x_values_to_focus_on) == 0:
        return image
    
    # Draw a highlight color on the x values to focus on
    mask = image.convert('RGBA').copy()
    mask_draw = ImageDraw.Draw(mask)
    #print(x_values_to_focus_on)
    
    # Iterate over the x values to highlight
    for x_value in x_values_to_focus_on:
        # Convert the bounding box to pixel coordinates
        try:
            x_value_bbox = all_x_values_bounding_boxes[x_value]
            x1 = int(x_value_bbox['x1'] + 2)
            y1 = int(x_value_bbox['y1'] + 2)
            x2 = int(x_value_bbox['x2'] - 2)
            y2 = int(x_value_bbox['y2'] - 2)
            mask_draw.rectangle(((x1, y1), (x2, y2)), fill=(255, 0, 0, 50))
        except Exception as e:
            print(f"Error: {e}")
        
    # Composite the overlay with the mask over the original image
    image_with_focused_x_values = Image.alpha_composite(image.convert('RGBA'), mask)
    return image_with_focused_x_values

def focus_on_y_values_with_highlight(image, y_values_to_focus_on, all_y_values_bounding_boxes):
    """
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
    """
    if len(all_y_values_bounding_boxes) == 0 or len(y_values_to_focus_on) == 0:
        return image
    
    # Draw a highlight color on the columns to focus on
    mask = image.convert('RGBA').copy()
    mask_draw = ImageDraw.Draw(mask)
    #print(y_values_to_focus_on)

    # Iterate over the y values to highlight
    for y_value in y_values_to_focus_on:
        # Convert the bounding box to pixel coordinates
        y_value_bbox = all_y_values_bounding_boxes[y_value]
        x1 = int(y_value_bbox['x1'] + 2)
        y1 = int(y_value_bbox['y1'] + 2)
        x2 = int(y_value_bbox['x2'] - 2)
        y2 = int(y_value_bbox['y2'] - 2)
        mask_draw.rectangle(((x1, y1), (x2, y2)), fill=(255, 0, 0, 50))     
        
    # Composite the overlay with the mask over the original image
    image_with_focused_y_values = Image.alpha_composite(image.convert('RGBA'), mask)
    return image_with_focused_y_values
