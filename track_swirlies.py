import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

def track_swirlies(observation, template, prev_swirlies, print_to_terminal=False):
    """
    Tracks swirlies in the given observation using template matching.
    
    Args:
        observation (numpy.ndarray): The current frame of the game.
        template (numpy.ndarray): The template image of the swirly to match.
        prev_swirlies (list): List of positions of swirlies in the previous frame.
        print_to_terminal (bool): Flag to control printing to the terminal.
    
    Returns:
        num_swirlies (int): The num_swirlies based on the presence of swirlies.
        current_swirlies (list): List of positions of swirlies in the current frame.
        collected_swirlies (int): Number of swirlies collected.
    """
    # Ensure observation is a valid NumPy array
    observation = np.array(observation, dtype=np.uint8)
    
    # Convert the observation to grayscale
    # gray_observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    # print(gray_observation.shape)
    # Convert the template to grayscale
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # print(gray_observation.shape)
    
    # observation needs to be ~400, 550
    # Perform template matching
    result = cv2.matchTemplate(observation, gray_template, cv2.TM_CCOEFF_NORMED)
    
    # Define a threshold for detecting the swirly
    threshold = 0.8
    
    # Find all locations where the match exceeds the threshold
    locations = np.where(result >= threshold)
    
    # Initialize num_swirlies and collected swirlies count
    num_swirlies = 0
    collected_swirlies = 0
    
    # Define the size of the box
    box_size = 7
    
    # Define the frame rectangle (e.g., 3% margin from each side)
    margin = 0.03
    frame_rect = (
        int(observation.shape[1] * margin),  # left
        int(observation.shape[0] * (margin + 0.02)),  # top
        int(observation.shape[1] * (1 - margin)),  # right
        int(observation.shape[0] * (1 - margin - 0.07))  # bottom
    )
    
    if print_to_terminal:
        # Draw the frame rectangle (for visualization)
        cv2.rectangle(observation, (frame_rect[0], frame_rect[1]), (frame_rect[2], frame_rect[3]), (255, 0, 0), 2)
    
    # Prepare bounding boxes and confidence scores for NMS
    boxes = []
    confidences = []
    
    for pt in zip(*locations[::-1]):
        box = [pt[0], pt[1], box_size, box_size]
        boxes.append(box)
        confidences.append(result[pt[1], pt[0]])
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    confidences = np.array(confidences)
    
    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), score_threshold=threshold, nms_threshold=0.3)
    
    # Initialize list of current swirlies
    current_swirlies = []
    
    # Check if any swirlies are on screen
    for i in indices:
        box = boxes[i]
        pt = (box[0], box[1])
        current_swirlies.append(pt)
        
        # Calculate the intersection area with the frame rectangle
        x_left = max(frame_rect[0], pt[0])
        y_top = max(frame_rect[1], pt[1])
        x_right = min(frame_rect[2], pt[0] + box_size)
        y_bottom = min(frame_rect[3], pt[1] + box_size)
        
        if x_right < x_left or y_bottom < y_top:
            intersection_area = 0
        else:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        swirly_area = box_size * box_size
        overlap_ratio = intersection_area / swirly_area
        
        # Check if the swirly is within the frame rectangle
        if overlap_ratio >= 0.75:
            # Draw a 24x24 pixel rectangle around the detected swirly (for visualization)
            if print_to_terminal:
                cv2.rectangle(observation, pt, (pt[0] + box_size, pt[1] + box_size), (0, 255, 0), 2)
                print(f"Swirly on screen at: {pt}")
        else:
            # Draw a 24x24 pixel rectangle around the detected swirly (for visualization)
            if print_to_terminal:
                cv2.rectangle(observation, pt, (pt[0] + box_size, pt[1] + box_size), (0, 0, 255), 2)
                print(f"Swirly off screen at: {pt}")
    
    # Calculate collected swirlies using the Hungarian algorithm
    if prev_swirlies:
        cost_matrix = np.zeros((len(prev_swirlies), len(current_swirlies)))
        for i, prev_pt in enumerate(prev_swirlies):
            for j, curr_pt in enumerate(current_swirlies):
                cost_matrix[i, j] = np.linalg.norm(np.array(prev_pt) - np.array(curr_pt))
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for i, prev_pt in enumerate(prev_swirlies):
            if i not in row_ind:
                # Check if the swirly was on screen and is now off screen
                if frame_rect[0] <= prev_pt[0] <= frame_rect[2] and frame_rect[1] <= prev_pt[1] <= frame_rect[3]:
                    collected_swirlies += 1

        for i, j in zip(row_ind, col_ind):
            # Check if the swirly was on screen and is now off screen
            if cost_matrix[i, j] < box_size:
                if frame_rect[0] <= prev_swirlies[i][0] <= frame_rect[2] and frame_rect[1] <= prev_swirlies[i][1] <= frame_rect[3]:
                    if not (frame_rect[0] <= current_swirlies[j][0] <= frame_rect[2] and frame_rect[1] <= current_swirlies[j][1] <= frame_rect[3]):
                        collected_swirlies += 1
    
    num_swirlies = len(current_swirlies)  # Number of detected swirlies
    # Print the number of swirlies detected
    if print_to_terminal:
        print(f"Number of swirlies detected: {num_swirlies}")
        print(f"Number of swirlies collected: {collected_swirlies}")
    
        # Display the observation with detected swirlies (for visualization)
        cv2.imshow("Detected Swirlies", observation)
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()
    
    return num_swirlies, current_swirlies, collected_swirlies

# Example usage
if __name__ == "__main__":
    # Load example observations (frames) and template image
    observations = [
        cv2.imread("test_images/test_image5.png"),        
        cv2.imread("test_images/test_image6.png")
    ]
    observations2 = [
        cv2.imread("test_images/test_image7.png"),
        cv2.imread("test_images/test_image8.png"),
        cv2.imread("test_images/test_image9.png"),
        cv2.imread("test_images/test_image10.png"),
        cv2.imread("test_images/test_image11.png")
    ]
    
    template = cv2.imread("image_templates/swirly.png")
    
    # Initialize previous swirlies list
    prev_swirlies = []
    
    # Loop through each observation and track swirlies
    for i, observation in enumerate(observations):
        gray_observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)

        reward, current_swirlies, collected_swirlies = track_swirlies(gray_observation, template, prev_swirlies, print_to_terminal=True)
        print(f"\nObservation {i+1}:")
        print(f"reward: {reward}")
        print(f"Current Swirlies: {len(current_swirlies)}")
        print(f"Collected Swirlies: {collected_swirlies}\n\n")
        
        # Update previous swirlies list
        prev_swirlies = current_swirlies
