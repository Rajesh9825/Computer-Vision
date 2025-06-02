import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image,ImageTk

# Dictionary to score PIL Image
images = {"Original":None,"Sketch":None}

def open_file():
    filepath = filedialog.askopenfilename()
    if not filepath:
        return
    img = cv2.imread(filepath)
    display_image(img,original=True)
    sketch_img = convert_to_sketch(img)
    display_image(sketch_img,original=False)


def convert_to_sketch(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    inverted_img = cv2.bitwise_not(gray_img)
    blurred_img = cv2.GaussianBlur(inverted_img,(21,21),sigmaX=0,sigmaY=0)
    inverted_blur_img = cv2.bitwise_not(blurred_img)
    sketch_img = cv2.divide(gray_img,inverted_blur_img,scale=256.0)
    return sketch_img


def display_image(img,original):
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) if original else img
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((300, 300))
    img_tk = ImageTk.PhotoImage(image=img_pil)

    # Store the PIL Image in the dictionary
    if original:
        images['Original'] = img_pil
    else:
        images['Sketch'] = img_pil
    
    label = original_image_label if original else sketch_image_label
    label.config(image= img_tk)
    label.image = img_tk


def save_sketch():
    if images['Sketch'] is None:
        messagebox.showerror("Error","No Sketch to save.")
        return 
    
    sketch_filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files","*.png")])
    if not sketch_filepath:
        return
    
    # Save the PIL image (sketch) to the file
    images["Sketch"].save(sketch_filepath,"PNG")
    messagebox.showinfo("Saved","Sketch Saved to {}".format(sketch_filepath))


app = tk.Tk()
app.title("Pencil Sketch Converter")

frame = tk.Frame(app)
frame.pack(pady=10,padx=10)

original_image_label = tk.Label(frame)
original_image_label.grid(row=0,column=0,padx=5,pady=5)
sketch_image_label =  tk.Label(frame)
sketch_image_label.grid(row=0,column=1,padx=5,pady=5)

btn_frame = tk.Frame(app)
btn_frame.pack(pady=10)

open_button = tk.Button(btn_frame,text="Open Image", command=open_file)
open_button.grid(row=0,column=0,padx=5)

save_button = tk.Button(btn_frame,text="Save Sketch",command=save_sketch)
save_button.grid(row=0,column=1,padx=5)

app.mainloop()

    