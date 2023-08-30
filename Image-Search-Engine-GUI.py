import tkinter as tk  # GUI library
from tkinter import filedialog  # File dialog module
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import random



# Load the YOLO object detection model
net = cv2.dnn.readNet('model\yolov3.weights', 'model\yolov3.cfg')
classes = open('model\coco.names').read().strip().split('\n')


def take_image():
    global image_path
    image_path = filedialog.askopenfilename()  # Open a file dialog to select an image

def take_dataset():
    global dataset_path
    dataset_path = filedialog.askdirectory()  # Open a file dialog to select an image

def detect_objects(image):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outs = net.forward(layer_names)

    boxes = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                width = int(detection[2] * image.shape[1])
                height = int(detection[3] * image.shape[0])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, width, height])
                class_ids.append(class_id)
    return boxes, class_ids
    

def draw_boxes(image, boxes, class_ids):
    for i, box in enumerate(boxes):
        x, y, w, h = box
        class_id = class_ids[i]
        color = (0, 255, 0)  # Green color for the rectangle
        class_name = classes[class_id]

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


def show_images(unique_labels, dataset_folder, image_display_frame,  num_images_per_class=1, ):

    
    for widget in image_display_frame.winfo_children():
        widget.destroy()

    for label in unique_labels:
        label_folder = os.path.join(dataset_folder, label)
        image_files = [f for f in os.listdir(label_folder) if f.endswith(".jpg")]

        if image_files:
            label_heading = tk.Label(image_display_frame, text=label)
            label_heading.pack()

            # Randomly select num_images_per_class images from the list
            selected_image_files = random.sample(image_files, num_images_per_class)

            for image_file in selected_image_files:
                image_path = os.path.join(label_folder, image_file)
                img = Image.open(image_path)
                img.thumbnail((100, 100))  # Resize the image to fit within the frame
                img_tk = ImageTk.PhotoImage(img)

                label = tk.Label(image_display_frame, image=img_tk)
                label.image = img_tk  # Keep a reference to prevent the image from being garbage collected
                label.pack(side=tk.LEFT)

            # Update the GUI window to display the images
            root.update()


def Search():
    global image_path
    image = cv2.imread(image_path)
    
    boxes, class_ids = detect_objects(image)
    image_with_boxes = draw_boxes(image, boxes, class_ids)
    cv2.imshow("Detected Objects", image_with_boxes)
    
    unique_class_ids = list(set(class_ids))  # Get unique class IDs

    unique_labels = [classes[class_id] for class_id in unique_class_ids]  # Get unique class labels

    # Update result label to display unique class labels
    result_label.config(text="\n".join(unique_labels))


    show_images(unique_labels, dataset_path, image_display_frame,int(image_num_in.get()))





# Create the main GUI window
root = tk.Tk()  # Initialize the main GUI window
root.title("Image Analysis")  # Set the title of the GUI window
root.geometry(f"{400}x{300}")  # Set the dimensions of the GUI window
root.configure(bg='light blue')  # Set the background color of the GUI window

# Create GUI components
upload_button = tk.Button(root, text="Browse Image", command=take_image)  # Create a button to upload an image
upload_button.pack(pady=10)  # Add padding and display the button

# Space element
label = tk.Label(root, text="")  # Create a space label
label.configure(bg='light blue')  # Set background color
label.pack()  # Display the space label

# Button to browse and select images
browse_button = tk.Button(root, text="Browse dataset", command=take_dataset)  # Create a button to browse and select images
browse_button.configure(bg='light blue')  # Set background color
browse_button.pack()  # Display the button

# Space element
label = tk.Label(root, text="")  # Create a space label
label.configure(bg='light blue')  # Set background color
label.pack()  # Display the space label

label = tk.Label(root, text="Number of Images:")  # Create a label for the number of images input
label.configure(bg='light blue')  # Set background color
label.pack()  # Display the label

image_num_in = tk.Entry(root)  # Create an input field for the number of clusters
image_num_in.pack()  # Display the input field

# Button to initiate the clustering process
button = tk.Button(root, text="    -> Search <-      ", command = Search)  # Create a button to initiate image display
button.configure(bg='cyan2')  # Set background color
button.pack()  # Display the button

result_label = tk.Label(root, text="", justify="left")  # Create a label to display results
result_label.pack(pady=10)  # Display the label with padding

image_display_frame = tk.Frame(root)
image_display_frame.pack()

# Start the GUI event loop
root.mainloop()  # Start the GUI event loop to handle user interactions
