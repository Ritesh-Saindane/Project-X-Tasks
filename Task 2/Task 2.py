import numpy as np
import cv2


# Function to Check if a pixel is black
def is_black(pixel):
    return pixel < 50


# Find the column with the least black pixels
def find_col(image):
    last_black_count = -1
    last_col = -1
    for x in range(image.shape[1]):
        black_count = 0
        for y in range(image.shape[0]):
            if is_black(image[y][x]):
                black_count += 1
        if black_count < last_black_count:
            return last_col
        last_black_count = black_count
        last_col = x
    return -1


# Find the row with the least black pixels
def find_row(image):
    last_black_count = -1
    last_row = -1
    for y in range(image.shape[0]):
        black_count = 0
        for x in range(image.shape[1]):
            if is_black(image[y][x]):
                black_count += 1
        if black_count < last_black_count:
            return last_row
        last_black_count = black_count
        last_row = y
    return -1


# Get the heights of the vertical blocks in col
def colHeights(image, col_x):
    heights = []
    start = -1
    end = -1
    for y in range(image.shape[0]):
        if is_black(image[y, col_x]):
            if start == -1:
                start = y
                end = y
            else:
                end = y
        else:
            if start != -1:
                height = end - start
                heights.append(height)
                start = -1
                end = -1
    if len(heights) == 6:
        return heights


# Get the widths of the horizontal blocks in row
def rowWidths(image, row_y):
    widths = []
    start = -1
    end = -1
    for x in range(image.shape[1]):
        if is_black(image[row_y, x]):
            if start == -1:
                start = x
                end = x
            else:
                end = x
        else:
            if start != -1:
                width = end - start
                widths.append(width)
                start = -1
                end = -1
    if len(widths) == 5:
        return widths[1:]  # Exclude the first block, which represents numbers


# Check if there is a larger block
def has_larger_block(widths):
    return (np.max(widths) - np.min(widths) > 10)



NUMBERS = {0: "Ace", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6"}
SUITS = {0: "Hearts", 1: "Clubs", 2: "Spades", 3: "Diamonds"}



def get_card_number(widths):
    i = widths.index(max(widths))
    return NUMBERS[i]



def get_card_suit(widths):
    i = widths.index(max(widths))
    return SUITS[i]



def rotate_image(image):
    new_image = np.zeros_like(image)
    height = image.shape[0]
    width = image.shape[1]
    for y in range(height):
        for x in range(width):
            new_image[height - 1 - y, width - 1 - x] = image[y, x]
    return new_image



def get_card_value(image):
    padding = 25
    image = image[padding:(image.shape[0] - padding), padding:(image.shape[1] - padding)]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    col_x = find_col(image)
    row_y = find_row(image)

    vertical_heights = colHeights(image, col_x)
    horizontal_widths = rowWidths(image, row_y)

    if not has_larger_block(vertical_heights):
        image = rotate_image(image)
        col_x = find_col(image)
        row_y = find_row(image)
        vertical_heights = colHeights(image, col_x)
        horizontal_widths = rowWidths(image, row_y)

    result = f"{get_card_number(vertical_heights)} of {get_card_suit(horizontal_widths)}"
    return result



for i in range(1, 3):
    for j in range(1, 4):
        img_path = f"tc{i}-{j}.png"
        img = cv2.imread(img_path)
        if img is not None:
            value = get_card_value(img)
            print(f"\n tc{i}-{j}.png : {value}")
            cv2.imshow(f"tc{i}-{j}.png", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Failed to load image: tc{i}-{j}.png")
