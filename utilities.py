#IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import PIL
import os
import sys
import glob
import random

from pprint import pprint
from ipywidgets import Video

from PIL import Image as PILImage
from PIL.ExifTags import TAGS

from IPython.display import display


#UTILITIES FOR PRINTING
def imshow(a, size=1.0):
    a = a.clip(0, 255).astype("uint8")
    
    if size != 1.0:
        new_dim = (int(a.shape[1] * size), int(a.shape[0] * size))
        a = cv2.resize(a, new_dim, interpolation=cv2.INTER_AREA)
    
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    display(PIL.Image.fromarray(a))
    
#PROPER FUNCTIONS BELOW 
#----------------------------------------------------------------------------------------

def sort_corners(corners):
    corners = np.array(corners)
    
    sorted_by_y = sorted(corners, key=lambda x: x[1])
    top_points = sorted(sorted_by_y[:2], key=lambda x: x[0])  # Top points sorted by x
    bottom_points = sorted(sorted_by_y[2:], key=lambda x: x[0])  # Bottom points sorted by x

    top_left, top_right = top_points
    bottom_left, bottom_right = bottom_points
    
    return top_left, top_right, bottom_right, bottom_left


def detect_board(state):
    gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1) #Make the edges thicker
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours

    largest_rect = None
    max_area = 0

    state_copy = state.copy()
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)  
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the polygon has 4 points
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            cv2.drawContours(state_copy, [approx], -1, (0, 255, 0), 2)
            if area > max_area: # Check if the area is the largest, if it is it should be the board
                largest_rect = approx
                max_area = area
    
    if largest_rect is not None:
        largest_rect_points = largest_rect.reshape(-1, 2)  

        n= 70
        width = 36 * n  
        height = 29.7 * n  # 

        dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

        top_left, top_right, bottom_right, bottom_left = sort_corners(largest_rect_points)
                 
        M = cv2.getPerspectiveTransform(np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32"), dst_points)

        return M, width, height
        
    return None, None, None


# ------------------------------------------------------------------------

def detect_dice(frame, template):
    # template is provided as a read grayscale image
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # calculate the center of the dice and the bounding box (top left and bottom right corners)

    top_left = max_loc
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    dice_center = (top_left[0] + template.shape[1] // 2, top_left[1] + template.shape[0] // 2)
    
    # return the center of the dice and the bounding box coords

    return dice_center, top_left, bottom_right


def count_dice_pips(frame, dice_center):
    # crop the image to get the dice top region

    top_left = (dice_center[0] - 25, dice_center[1] - 25)
    bottom_right = (dice_center[0] + 25, dice_center[1] + 25)

    crop_top_left = (max(0, top_left[0]), max(0, top_left[1]))
    crop_bottom_right = (min(frame.shape[1], bottom_right[0]), min(frame.shape[0], bottom_right[1]))

    # extract the cropped region of the dice

    only_dice = frame[crop_top_left[1]:crop_bottom_right[1], crop_top_left[0]:crop_bottom_right[0]]        
    only_dice = cv2.resize(only_dice, (100, 100))
    only_dice = cv2.GaussianBlur(only_dice, (3,3), 0)
    only_dice_gray = cv2.cvtColor(only_dice, cv2.COLOR_BGR2GRAY)
    
    # hough circle transform to detect the dice pips
    circles = cv2.HoughCircles(only_dice_gray, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1,   
                               minDist=10, 
                               param1=50, 
                               param2=12, 
                               minRadius=3, 
                               maxRadius=10)

    if circles is not None:
        pips = len(circles[0])
        if pips > 6:
            pips = 6
    else:
        pips = 0

    return pips

# ------------------------------------------------------------------------



def board_corners(board):
    #Caculate the corners of the grid knowing it is at the center of the board
    #Utilize sam scaling factor as the board knowing each tile is 2x2 in real life
    
    grid_width = 1
    grid_height = 1

    tile_size = 20 * 70

    height, width = board.shape[:2]
    center_x, center_y = width // 2, height // 2

    offset_x = center_x - (grid_width * tile_size) // 2
    offset_y = center_y - (grid_height * tile_size) // 2

    top_left = (offset_x, offset_y)
    top_right = (offset_x + grid_width * tile_size, offset_y)
    bottom_right = (offset_x + grid_width * tile_size, offset_y + grid_height * tile_size)
    bottom_left = (offset_x, offset_y + grid_height * tile_size)
    
    return top_left, top_right, bottom_right, bottom_left



def get_score(board):
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray) #Obtain grayscale image with equalized histogram
    
    margin = 2*2*70
    
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=8, param2=25, minRadius=62, maxRadius=80)
    
    green_points = 0
    red_points = 0
    yellow_points = 0
    blue_points = 0
    
    green_circles=[]
    red_circles=[]
    yellow_circles=[]
    blue_circles=[]
    
    board_copy = board.copy()
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            mask = np.zeros_like(clahe_img, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            masked = cv2.bitwise_and(board, board, mask=mask)
            
            circle_brightness = cv2.mean(clahe_img, mask=mask)[0] #Calculate the mean brightness of the circle
            
            outer_radius = r + 30
            large_mask = np.zeros_like(clahe_img, dtype=np.uint8)
            cv2.circle(large_mask, (x, y), outer_radius, 255, -1)
            
            surrounding_mask = cv2.subtract(large_mask, mask)
            background_brightness = cv2.mean(clahe_img, mask=surrounding_mask)[0] #Calculate the mean brightness of the neighborhood
            
            contrast = background_brightness - circle_brightness #Calculate the contrast between the circle and the background
            
            mean_color = cv2.mean(board, mask=mask)[:3]
            mean_color = tuple(map(int, mean_color))

            if contrast > 57: #If the contrast between the circle and the background is high
                if x < margin: 
                    blue_points += 1
                    blue_circles.append((x, y, r))

                elif x >= board.shape[1] - margin:
                    green_points += 1
                    green_circles.append((x, y, r))

                elif y < margin:
                    yellow_points += 1
                    yellow_circles.append((x, y, r))

                elif y >= board.shape[0] - margin:
                    red_points += 1
                    red_circles.append((x, y, r))

    #Return the number of points and circles for annotating on board
    return (green_points, green_circles), (red_points, red_circles), (yellow_points, yellow_circles), (blue_points, blue_circles)


def calculate_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    return None


def calculate_base(contour):
    lowest_point = max(contour, key=lambda x: x[0][1])
    lowest_point[0][1]-=60
    return tuple(lowest_point[0])


def get_position(board):
    corners = board_corners(board)
    
    centroid = np.mean(corners, axis=0)
    scale_factor = 1
    
    expanded_corners = (corners - centroid) * scale_factor + centroid
    
    mask = np.zeros_like(board, dtype=np.uint8)
    roi_corners = np.array(expanded_corners, dtype=np.int32)
    cv2.fillPoly(mask, [roi_corners], (255, 255, 255))
    
    masked = cv2.bitwise_and(board, mask)

    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    
    color_ranges = {
    "red": [(150, 100, 100), (178.5, 255, 255)],
    "yellow": [(15, 138, 128), (30, 238, 255)],
    "green": [(30, 25, 63), (80, 200, 200)],
    "blue": [(80, 60, 60), (150, 255, 255)],
    }
    
    masks = {}
    for color, (lower, upper) in color_ranges.items(): #Get a mask for each color
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        masks[color] = cv2.inRange(hsv, lower, upper)
    
    combined_mask = np.zeros_like(masks["red"])

    for mask in masks.values():
        combined_mask = cv2.bitwise_or(combined_mask, mask) #Combine all masks
     
    combined_mask = cv2.dilate(combined_mask, np.ones((3, 3), np.uint8), iterations=3)
    
    segmented = cv2.bitwise_and(masked, masked, mask=combined_mask)
    
    edges = cv2.Canny(combined_mask, 50, 150) #Detect edges
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    convex_hulls = [cv2.convexHull(cnt) for cnt in contours]
    filtered_hulls = [cnt for cnt in convex_hulls if cv2.contourArea(cnt) > 5000] #Filter out small contours
    
    centroids = [calculate_centroid(cnt) for cnt in filtered_hulls]
    threshold = 100
    merged_countours = []
    for i, cnt in enumerate(filtered_hulls): #Merge contours that are close to each other
        centroid = centroids[i]
        if centroid is None:
            continue
        merged = False
        for merged_cnt in merged_countours:
            if calculate_centroid(merged_cnt) and math.dist(centroid, calculate_centroid(merged_cnt)) < threshold:
                merged_cnt = np.concatenate([merged_cnt, cnt], axis=0)
                merged = True
                break
        
        if not merged:
            merged_countours.append(cnt)
    
    filtered_hulls = merged_countours
    
    board_copy = board.copy()
    cv2.drawContours(board_copy, filtered_hulls, -1, (0, 255, 0), 2)
    for cnt in filtered_hulls:
        cv2.putText(board_copy, f"{cv2.contourArea(cnt):.2f}", calculate_base(cnt), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
    green_position = None
    red_position = None
    yellow_position = None
    blue_position = None
    
    for cnt in filtered_hulls: #For each contour, calculate the base and the mean color
        if cnt is not None:
            base = calculate_base(cnt)
            cv2.circle(board_copy, base, 10, (0, 0, 255), -1)
            mean_colors = []
            
            for color in color_ranges.keys():
                color_mask = masks[color]
                mask = np.zeros(color_mask.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                
                mean_color = cv2.mean(color_mask, mask=mask)
                mean_colors.append(mean_color[0])
                
            max_indx = np.argmax(mean_colors) #Get the color with the highest mean value of overlapping pixels
            if max_indx == 0:
                red_position = base
            elif max_indx == 1:
                yellow_position = base
            elif max_indx == 2:
                green_position = base
            elif max_indx == 3:
                blue_position = base
            
    return green_position, red_position, yellow_position, blue_position


def get_coord(board, position):
    #Calculates the coordinates on the grid of the pawn based on its x,y position and upper left corner of the grid
    if position is None:
        return None
    corners = board_corners(board)
    upper_left = corners[0]
    return ((position[0] - upper_left[0])//140, (position[1] - upper_left[1])//140)


def get_state(board, prev_state=None):
    #Updates the state of the game based on the board
    
    if prev_state is not None:
        prev_green, prev_red, prev_yellow, prev_blue = prev_state
    else:
        prev_green, prev_red, prev_yellow, prev_blue = None, None, None, None
    
    #Read the positions, coordinates and scores of the pawns
    green_pos, red_pos, yellow_pos, blue_pos = get_position(board)
    green_coord, red_coord, yellow_coord, blue_coord = get_coord(board, green_pos), get_coord(board, red_pos), get_coord(board, yellow_pos), get_coord(board, blue_pos)
    green_score, red_score, yellow_score, blue_score = get_score(board)
    
    green_score, green_circles = green_score
    red_score, red_circles = red_score
    yellow_score, yellow_circles = yellow_score
    blue_score, blue_circles = blue_score
    
    if green_coord is None:
        green_coord = prev_green["coord"]
        green_pos = prev_green["position"]
    if red_coord is None:
        red_coord = prev_red["coord"]
        red_pos = prev_red["position"]
    if yellow_coord is None:
        yellow_coord = prev_yellow["coord"]
        yellow_pos = prev_yellow["position"]
    if blue_coord is None:
        blue_coord = prev_blue["coord"]
        blue_pos = prev_blue["position"]
    
    
    green ={
        "position": green_pos,
        "coord": green_coord,
        "score": green_score,
        "circles": green_circles
    }
    
    red ={
        "position": red_pos,
        "coord": red_coord,
        "score": red_score,
        "circles": red_circles
    }
    
    yellow ={
        "position": yellow_pos,
        "coord": yellow_coord,
        "score": yellow_score,
        "circles": yellow_circles
    }
    
    blue ={
        "position": blue_pos,
        "coord": blue_coord,
        "score": blue_score,
        "circles": blue_circles
    }
    
    return green, red, yellow, blue



def annotate_board(board, state):
    #Mark all the pawns coordinates and detected coins on the board
    
    green, red, yellow, blue = state
    
    corners = board_corners(board)
    upper_left = corners[0]
    
    green_x = upper_left[0] + green["coord"][0] * 140
    green_y = upper_left[1] + green["coord"][1] * 140
    
    red_x = upper_left[0] + red["coord"][0] * 140
    red_y = upper_left[1] + red["coord"][1] * 140
    
    yellow_x = upper_left[0] + yellow["coord"][0] * 140
    yellow_y = upper_left[1] + yellow["coord"][1] * 140
    
    blue_x = upper_left[0] + blue["coord"][0] * 140
    blue_y = upper_left[1] + blue["coord"][1] * 140

    #Knowing upper left corner of the board we can annotate the square with each pawn
    cv2.rectangle(board, (green_x, green_y), (green_x + 140, green_y + 140), (0, 255, 0), 5)
    cv2.rectangle(board, (red_x, red_y), (red_x + 140, red_y + 140), (0, 0, 255), 5)
    cv2.rectangle(board, (yellow_x, yellow_y), (yellow_x + 140, yellow_y + 140), (0, 255, 255), 5)
    cv2.rectangle(board, (blue_x, blue_y), (blue_x + 140, blue_y + 140), (255, 0, 0), 5)
    
    #Annotate the circles on the board with appropriate colors
    for color, circles in zip([(0,255,0), (0,0,255), (0,255,255), (255,0,0)] ,
                               [green["circles"], red["circles"], yellow["circles"], blue["circles"]]):
            for circle in circles:
                cv2.circle(board, (circle[0], circle[1]), circle[2], color, 5)
                

def unwarp_coordinates(coords, M_inv):
    #Used for transofrming the coordinates on the board to the ones on the frame itself
    
    coords = np.array(coords, dtype="float32")
    coords = np.array([coords])  # Shape it for cv2.perspectiveTransform
    unwarped_coords = cv2.perspectiveTransform(coords, M_inv)
    return unwarped_coords[0].astype(int)


def annotate_frame(frame, prev_frame, dice_template, lk_params, p0, dice_center, dice_top_left, dice_bottom_right, number_of_pips, board, state, M_inv):

    # dice tracking logic ---------------------------------------------------

    current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, p0, None, **lk_params)

    is_rolling = False

    if st[0][0] == 1:

        motion = np.linalg.norm(p1- p0)
        if motion < 1.5:
            p1 = p0
        
        else:
            is_rolling = True

        number_of_pips = count_dice_pips(frame, dice_center)
        dice_center = int(p1[0][0][0]), int(p1[0][0][1])  

    else:
        dice_center, dice_top_left, dice_bottom_right = detect_dice(frame, dice_template)
        number_of_pips = count_dice_pips(frame, dice_center)

    # place the tracking dot on the dice 

    cv2.circle(frame, dice_center, 5, (0, 0, 255), -1)
    prev_frame = current_frame.copy()
    p0 = np.array([[dice_center]], dtype=np.float32)

    # i need to return the updated frame, prev_frame, p0, dice_center, dice_top_left, dice_bottom_right
    # in order to continue tracking the dice in the next frame
    # -------------------------------------------------------------------------
    
    green, red, yellow, blue = state
    
    corners = board_corners(board)
    upper_left = corners[0]
    
    #Calculate every corner of the tile with a pawn on the board
    green_x = upper_left[0] + green["coord"][0] * 140
    green_y = upper_left[1] + green["coord"][1] * 140
    green_corners = [[green_x, green_y],[green_x+140, green_y], [green_x + 140, green_y + 140], [green_x, green_y + 140]]
    
    red_x = upper_left[0] + red["coord"][0] * 140
    red_y = upper_left[1] + red["coord"][1] * 140
    red_corners = [[red_x, red_y],[red_x+140, red_y], [red_x + 140, red_y + 140], [red_x, red_y + 140]]
    
    yellow_x = upper_left[0] + yellow["coord"][0] * 140
    yellow_y = upper_left[1] + yellow["coord"][1] * 140
    yellow_corners = [[yellow_x, yellow_y],[yellow_x+140, yellow_y], [yellow_x + 140, yellow_y + 140], [yellow_x, yellow_y + 140]]
    
    blue_x = upper_left[0] + blue["coord"][0] * 140
    blue_y = upper_left[1] + blue["coord"][1] * 140
    blue_corners = [[blue_x, blue_y],[blue_x+140, blue_y], [blue_x + 140, blue_y + 140], [blue_x, blue_y + 140]]
    
    #Unwarp the coordinates of the corners of tiles with the pawn on the board to annotate them on the frame
    green_unwarped = unwarp_coordinates(green_corners, M_inv)
    green_unwarped = np.array(green_unwarped).reshape(-1, 1, 2)
        
    red_unwarped = unwarp_coordinates(red_corners, M_inv)
    red_unwarped = np.array(red_unwarped).reshape(-1, 1, 2)
        
    yellow_unwarped = unwarp_coordinates(yellow_corners, M_inv)
    yellow_unwarped = np.array(yellow_unwarped).reshape(-1, 1, 2)
        
    blue_unwarped = unwarp_coordinates(blue_corners, M_inv)
    blue_unwarped = np.array(blue_unwarped).reshape(-1, 1, 2)
    
    #Use polylines as tiles may not be perfectly square
    cv2.polylines(
            frame,
            [green_unwarped],
            isClosed=True,
            color = (0, 255, 0),
            thickness = 3
        )
        
    cv2.polylines(
            frame,
            [red_unwarped],
            isClosed=True,
            color = (0, 0, 255),
            thickness = 3
        )
        
    cv2.polylines(
            frame,
            [yellow_unwarped],
            isClosed=True,
            color = (0, 255, 255),
            thickness = 3
        )
        
    cv2.polylines(
            frame,
            [blue_unwarped],
            isClosed=True,
            color = (255, 0, 0),
            thickness = 3
        )
    
    #Annotate the circles on the frame with appropriate colors
    for color, circles in zip([(0,255,0), (0,0,255), (0,255,255), (255,0,0)] ,
                               [green["circles"], red["circles"], yellow["circles"], blue["circles"]]):
            for circle in circles:
                unwarped_center = unwarp_coordinates([circle[:2]], M_inv)

                #Random point needed as the circles may change radius after unwarping them 
                random_point = (circle[0] + circle[2], circle[1])
                unwarped_random = unwarp_coordinates([random_point], M_inv)
                
                new_radius = math.dist(unwarped_center[0], unwarped_random[0])
                new_radius = int(new_radius)
                
                cv2.circle(frame, (unwarped_center[0][0], unwarped_center[0][1]), new_radius, color, 2)
    
    return prev_frame, p0, dice_center, dice_top_left, dice_bottom_right, number_of_pips, is_rolling
                
            
            
def annotate_state(canvas, frame, state):
    #Used to write the state of the game on the canvas
    #Annotates the score and position of each pawn
    
    green, red, yellow, blue = state
    
    cv2.putText(
        canvas,
        "GAME STATE: ",
        (frame.shape[1] + 200, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA
        )
    cv2.putText(
        canvas,
        f'Green on {green["coord"]}; SCORE: {green["score"]}/8',
        (frame.shape[1] + 100, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (34, 139, 34),
        2,
        cv2.LINE_AA
    )
        
    cv2.putText(
        canvas,
        f'Red on {red["coord"]}; SCORE: {red["score"]}/8',
        (frame.shape[1] + 100, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 139),
        2,
        cv2.LINE_AA
    )
        
    cv2.putText(
        canvas,
        f'Blue on {blue["coord"]}; SCORE: {blue["score"]}/8',
        (frame.shape[1] + 100, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (139, 0, 0),
        2,
        cv2.LINE_AA
    )
        
    cv2.putText(
        canvas,
        f'Yellow on {yellow["coord"]}; SCORE: {yellow["score"]}/8',
        (frame.shape[1] + 100, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 204, 204),
        2,
        cv2.LINE_AA
    )
    

def annotate_events(canvas, frame, state, number_of_pips, is_rolling):
    #Used to write the events of the game on the canvas
    
    green, red, yellow, blue = state
    
    special_tiles = [
        (4, 0),
        (0, 1),
        (5, 2),
        (9, 2),
        (3, 4),
        (6, 4),
        (8, 6),
        (3, 7),
        (0, 8),
        (5, 9)
    ]
    
    cv2.putText(
        canvas,
        "GAME EVENTS: ",
        (frame.shape[1] + 180, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA
        )
    
    #Annotation of the dice roll
    message = f"Dice rolled: {number_of_pips}"

    if is_rolling:
        message += " (Rolling...)"
    
    cv2.putText(
        canvas,
        message,
        (frame.shape[1] + 180, 290),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (50, 50, 50),
        2,
        cv2.LINE_AA
    )

    #Annotation of the special tiles
    on_special_tile = []
    
    if green["coord"] in special_tiles:
        on_special_tile.append("Green")
    if red["coord"] in special_tiles:
        on_special_tile.append("Red")
    if yellow["coord"] in special_tiles:
        on_special_tile.append("Yellow")
    if blue["coord"] in special_tiles:
        on_special_tile.append("Blue")
        
    green_color = (34, 139, 34) if "Green" in on_special_tile else (169, 169, 169)
    red_color = (0, 0, 139) if "Red" in on_special_tile else (169, 169, 169)
    yellow_color = (0, 204, 204) if "Yellow" in on_special_tile else (169, 169, 169)
    blue_color = (139, 0, 0) if "Blue" in on_special_tile else (169, 169, 169)
    
    cv2.putText(
        canvas,
        "On special tile: ",
        (frame.shape[1] + 180, 335),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (50, 50, 50),
        2,
        cv2.LINE_AA
        )
        
    cv2.putText(
        canvas,
        "Green",
        (frame.shape[1] + 200, 370),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        green_color,
        2,
        cv2.LINE_AA
        )
        
    cv2.putText(
        canvas,
        "Red",
        (frame.shape[1] + 300, 370),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        red_color,
        2,
        cv2.LINE_AA
        )
        
    cv2.putText(
        canvas,
        "Yellow",
        (frame.shape[1] + 200, 400),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        yellow_color,
        2,
        cv2.LINE_AA
        )
        
    cv2.putText(
        canvas,
        "Blue",
        (frame.shape[1] + 300, 400),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        blue_color,
        2,
        cv2.LINE_AA
        )
    
    #Annotation of the event of "Game Starts Now"
    if yellow["coord"] == (0,0) and green["coord"] == (9, 0) and red["coord"] == (9, 9) and blue["coord"] == (0, 9):
        if yellow["score"] == 0 and green["score"] == 0 and red["score"] == 0 and blue["score"] == 0:
                cv2.putText(
                    canvas,
                    "GAME STARTS NOW ",
                    (frame.shape[1] + 160, 460),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 140, 255),
                    2,
                    cv2.LINE_AA
                )
    
    #Detects when game ends and annotates the winner
    if yellow["score"] == 8 or green["score"] == 8 or red["score"]== 8 or blue["score"] == 8:
        cv2.putText(
            canvas,
            "WINNER: ",
            (frame.shape[1] + 160, 510),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 140, 255),
            2,
            cv2.LINE_AA
        )
        if yellow["score"] == 8:
            winner = "Yellow"
            color = (0, 204, 204)
        elif green["score"] == 8:
            winner = "Green"
            color = (34, 139, 34)
        elif red["score"] == 8:
            winner = "Red"
            color = (0, 0, 139)
        elif blue["score"] == 8:
            winner = "Blue"
            color = (139, 0, 0)
            
        cv2.putText(
        canvas,
        winner,
        (frame.shape[1] + 300, 510),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA
        )


def annotate_debug(canvas, frame, frame_count):
    #Used to write the current frame number on the canvas for easier debugging
    
    cv2.putText(
            canvas,
            f"DEBUG  FRAME: {frame_count}",
            (frame.shape[1] + 100, 1050),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (50,205,50),
            2,
            cv2.LINE_AA
    )
    
def annotate_score_change(canvas, frame, got_coin, current_frame):
    #Annotate event of getting a coin
    colors = {
        "Green": (34, 139, 34),
        "Red": (0, 0, 139),
        "Yellow": (0, 204, 204),
        "Blue": (139, 0, 0)
    }
    
    for color in got_coin:
        if current_frame - got_coin[color] < 50: #The annotation will last for 50 frames
            cv2.putText(
                canvas,
                f"{color}",
                (frame.shape[1] + 160, 510),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                colors[color],
                2,
                cv2.LINE_AA
            )
            
            cv2.putText(
                canvas,
                "GOT COIN",
                (frame.shape[1] + 250, 510),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 140, 255),
                2,
                cv2.LINE_AA
            )

    