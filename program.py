import cv2
import joblib
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow import keras
import matplotlib.pyplot as plt


from kivy.app import App
from kivy.lang import Builder
from kivy.config import Config
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.core.image import Image as CoreImage
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

import tkinter as tk
from tkinter import filedialog


class WindowManager(ScreenManager):
    def __init__(self, **kwargs):
        super(WindowManager, self).__init__(**kwargs)
        self.main_window = MainWindow()
        self.second_window = SecondWindow()
        self.add_widget(self.main_window)
        self.add_widget(self.second_window)
        
    def main_to_second(self):
        pic = self.main_window.main_panel.pil_image
        if(pic is None):
            return
        
        # Convert from Pillow to OpenCV
        open_cv_image = np.array(pic) 
        # RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        
        app = App.get_running_app()
        app.root.current = "second_window"
        app.root.transition.direction = 'left'
        self.second_window.second_panel.slider.value = 75
        self.second_window.second_panel.switch.active = False
        self.second_window.second_panel.cv2_image = open_cv_image
        self.second_window.second_panel.filter_and_plot(open_cv_image, 75)


class MainWindow(Screen):
    def __init__(self, **kwargs):
        super(MainWindow, self).__init__(**kwargs)
        self.main_panel = MainPanel()
        self.add_widget(self.main_panel)


class SecondWindow(Screen):
    def __init__(self, **kwargs):
        super(SecondWindow, self).__init__(**kwargs)
        self.second_panel = SecondPanel()
        self.add_widget(self.second_panel)
        self.second_panel._plot_init()


class SecondPanel(Widget):
    plot_field = ObjectProperty(None)
    slider = ObjectProperty(None)
    slider_value = ObjectProperty(None)
    
    switch = ObjectProperty(None)
    vertical_button = ObjectProperty(None)
    horizontal_button = ObjectProperty(None)
    
    model_label = StringProperty("NeuralNetwork.h5")
    
    def __init__(self, **kwargs):
        super(SecondPanel, self).__init__(**kwargs)
        
        try:
            self.model = keras.models.load_model("models/NeuralNetwork.h5")
            #self.model = joblib.load("models/ForestClassifier.sav")
        except Exception as e:
            show_error("Error while loading model \n" + e)
            exit()
        
        self.cv2_image = None
        self.preprocessed_digits = None
        
    def _plot_init(self):
        self.ids.plot_field.clear_widgets()
        plt.clf()
        self.ids.plot_field.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        
    def filter_and_plot(self, photo, threshold):
        show_greyscale = ""
        if(self.switch.active):
            if(self.vertical_button.state == "down"):
                show_greyscale = "Vertical"
            elif(self.horizontal_button.state == "down"):
                show_greyscale = "Horizontal"
        else:
            self.vertical_button.state = "normal"
            self.horizontal_button.state = "normal"
            show_greyscale = "None"
        self.ids.plot_field.clear_widgets()
        plt.clf()
        self.preprocessed_digits = filter_and_plot(photo, threshold, show_greyscale)
        self.ids.plot_field.add_widget(FigureCanvasKivyAgg(plt.gcf()))
    
    def validate_input(self):
        
        if(self.slider_value.text.isnumeric()):
            value = int(self.slider_value.text)
        else:
            self.slider_value.text = str(self.slider.value)
            return
            
        if(value >= self.slider.min and value <= self.slider.max):
            self.slider.value = value  
        else:
            self.slider_value.text = str(self.slider.value)
            return
        
    def toggle_clicked(self):
        if (self.switch.active):
            self.filter_and_plot(self.cv2_image, self.slider.value)
        else:
            self.vertical_button.state = "normal"
            self.horizontal_button.state = "normal"
           
    def analyze(self):
        if(self.cv2_image is None):
            return
        
        result_string = ""
        line_change = 0
        try:
            for line, x, digit in self.preprocessed_digits: 
                trs = digit.reshape(1, 28, 28, 1)
                prediction = np.argmax(self.model.predict(trs))
                if(line > line_change):
                    result_string = result_string[:-2]
                    line_change += 1
                    result_string += "\n"
                result_string += str(prediction) + ", "
            show_popup(result_string[:-2])
        except Exception as e:
            show_error("Model or scaler fail \n" + e)
            return
            

def show_file_dialog(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        
    return file_path


class MainPanel(Widget):
    
    image = ObjectProperty(None)
    drawing_field = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(MainPanel, self).__init__(**kwargs)
        
        self.second_screen = SecondPanel()
        
        self.previous_image = None
        self.pil_image = None
        
        self.move_speed = 5
        self.size_increase = 5
        
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down, on_key_up=self._on_keyboard_up)
        
        self.is_cropping = False
         
    def get_image_position(self):
        X_POS = self.ids.image.center_x - self.ids.image.norm_image_size[0] / 2
        Y_POS = self.ids.image.center_y - self.ids.image.norm_image_size[1] / 2
        
        return (X_POS, Y_POS)
              
    def crop(self):
        if(self.pil_image is None):
            self.drawing_field.crop_button.state = "normal"
            return 
        if (self.drawing_field.crop_button.state == "down"):
            width, height = Window.size
            self.drawing_field.selection.pos[0] = width * 0.7 * 0.5 - 50
            self.drawing_field.selection.pos[1] = height * 0.5 - 50
            
            self.drawing_field.selection.size[0] = 100
            self.drawing_field.selection.size[1] = 100
            return
            
        image_x, image_y = self.get_image_position()
        selection_x, selection_y = self.drawing_field.selection.pos[0], self.drawing_field.selection.pos[1]
        selection_size_x, selection_size_y = self.drawing_field.selection.size[0], self.drawing_field.selection.size[1]
        
        window_height = Window.size[1]
        
        # coordinate system root transformation from lower to upper right corner
        image_y = window_height - (image_y + self.ids.image.norm_image_size[1])
        selection_y = window_height - (selection_y + selection_size_y)
        
        image_scale = self.pil_image.size[0] / self.ids.image.norm_image_size[0]
        
        transformed_x = (selection_x - image_x) * image_scale
        transformed_y = (selection_y - image_y) * image_scale
        selection_size_x *= image_scale
        selection_size_y *= image_scale
          
        self.pil_image = self.pil_image.crop((transformed_x, transformed_y, transformed_x + selection_size_x, transformed_y + selection_size_y))
        
        self.display_pil_image()
        
        self.drawing_field.selection.size[0] = -1
        self.drawing_field.selection.size[1] = -1
        self.drawing_field.selection.pos[0] = -500
        self.drawing_field.selection.pos[1] = -500
        self.drawing_field.crop_button.state == "normal"
           
    def display_pil_image(self):
        data = BytesIO()
        self.pil_image.save(data, format='png')
        data.seek(0) 
        im = CoreImage(BytesIO(data.read()), ext='png')
        self.image.texture = im.texture 
        
    def load(self):
        file_name = (show_file_dialog("Select a file", (("Images", ".jpg .jpeg .png"), ("All files", "*"))))
        
        if(file_name == () or file_name == ""):
            return
        
        self.pil_image = Image.open(file_name)
        self.previous_image = self.pil_image
        self.display_pil_image()
        
    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down, on_key_up=self._on_keyboard_up)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        
        if keycode[1] == 'right':
            self.drawing_field.selection.size[0] += self.size_increase
        elif keycode[1] == 'left':
            self.drawing_field.selection.size[0] -= self.size_increase 
        elif keycode[1] == 'up':
            self.drawing_field.selection.size[1] += self.size_increase    
        elif keycode[1] == 'down':
            self.drawing_field.selection.size[1] -= self.size_increase
        
        elif keycode[1] == 'r':
            self.restore()
        elif keycode[1] == '1':
            self.pil_image = Image.open("photos/1.png")
            self.previous_image = self.pil_image
            self.display_pil_image()
        elif keycode[1] == '2':
            self.pil_image = Image.open("photos/2.png")
            self.previous_image = self.pil_image
            self.display_pil_image()
        elif keycode[1] == '3':
            self.pil_image = Image.open("photos/3.png")
            self.previous_image = self.pil_image
            self.display_pil_image()
        elif keycode[1] == '4':
            self.pil_image = Image.open("photos/4.png")
            self.previous_image = self.pil_image
            self.display_pil_image()
        elif keycode[1] == '5':
            self.pil_image = Image.open("photos/5.png")
            self.previous_image = self.pil_image
            self.display_pil_image()
        elif keycode[1] == '6':
            self.pil_image = Image.open("photos/6.png")
            self.previous_image = self.pil_image
            self.display_pil_image()
                
        elif keycode[1] in ['shift', 'rshift']:
            self.move_speed *= 5
            self.size_increase *= 5
            
        return True
            
    def _on_keyboard_up(self, keyboard, keycode):
        if keycode[1] in ['shift', 'rshift']:
            self.move_speed /= 5
            self.size_increase /= 5
        
        return True
    
    def restore(self):
        if(self.previous_image is not None):
            self.pil_image = self.previous_image
            self.display_pil_image()
        
    def rotate_left(self):
        if(self.pil_image is None):
            return
        
        self.pil_image = self.pil_image.rotate(90)
        self.display_pil_image()
        
    def rotate_right(self):
        if(self.pil_image is None):
            return
        
        self.pil_image = self.pil_image.rotate(-90)
        self.display_pil_image()
        
         
class DrawingField(Widget):
    
    selection = ObjectProperty(None)
    crop_button = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(DrawingField, self).__init__(**kwargs)
        
        self.move_speed = 5
        self.size_increase = 5

    def on_touch_down(self, touch):
        if(self.crop_button.state == "normal"):
            return
        
        self.selection.pos = touch.pos
        self.selection.pos[0] -= self.selection.size[0] / 2
        self.selection.pos[1] -= self.selection.size[1] / 2
        
        if touch.button == 'scrolldown':
            self.selection.size[0] -= self.size_increase
            self.selection.size[1] -= self.size_increase
            
        elif touch.button == 'scrollup':
            self.selection.size[0] += self.size_increase
            self.selection.size[1] += self.size_increase
        
    def on_touch_move(self, touch):
        if(self.crop_button.state == "normal"):
            return
        
        self.selection.pos = touch.pos
        self.selection.pos[0] -= self.selection.size[0] / 2
        self.selection.pos[1] -= self.selection.size[1] / 2


class LineRectangle(Widget):
    pass
   
    
class MyPopup(Popup):
    result_label = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super(MyPopup, self).__init__(**kwargs)
      
        
class ErrorPopup(Popup):
    error_label = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super(ErrorPopup, self).__init__(**kwargs)


class MainApp(App):

    def build(self):
        kv = Builder.load_file("design.kv")
        self.title = 'Handwritten equation solver'
        return kv


def filter_and_plot(image, threshold, show_greyscale="Vertical"):

    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    thresh_display = thresh.copy()
    color_display = image.copy()
    
    preprocessed_digits = []
    lines = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Box display
        cv2.rectangle(color_display, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        
        digit = thresh[y:y + h, x:x + w]
        
        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18, 18))
        
        # Padding the digit with 5 pixels of (zeros) in each side as in MNIST
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
        
        preprocessed_digits.append((x, y, w, h, padded_digit))
          
    # finding line splits    
    for x, y, w, h, _ in preprocessed_digits:
        collision = False
        for xl, yl, wl, hl, _ in preprocessed_digits:
            if(y + h < yl + hl and y + h > yl + hl * 0.25):
                collision = True
                break
        if (y + h not in lines and collision is False):
            lines.append(y + h)
                        
    for line in lines:
        cv2.line(color_display, (-5, line), (999999, line), (0, 0, 255), 2)
        
    result = []
    # assigning lines to digits
    for x, y, w, h, digit in preprocessed_digits:
        min_distance = 999999
        min_line = 0
        for index, line in enumerate(lines):
            if (line - (y + h) >= 0) and (line - (y + h) < min_distance):
                min_distance = (line - y)
                min_line = index
        result.append((len(lines) - min_line - 1, x, digit))
            
    result.sort(key=lambda x: (x[0], x[1]))

    if(show_greyscale == "Horizontal"):
        thresh_3_channel = cv2.cvtColor(thresh_display, cv2.COLOR_GRAY2BGR)
        numpy_horizontal = np.hstack((color_display, thresh_3_channel))
        plt.imshow(numpy_horizontal, cmap="gray")
    elif(show_greyscale == "Vertical"):
        thresh_3_channel = cv2.cvtColor(thresh_display, cv2.COLOR_GRAY2BGR)
        numpy_horizontal = np.vstack((color_display, thresh_3_channel))
        plt.imshow(numpy_horizontal, cmap="gray")  
    else:
        plt.imshow(color_display, cmap="gray")    
    
    plt.axis("off")
    
    return result
    
    
def show_popup(result):
    popupWindow = MyPopup()
    popupWindow.result_label.text = str(result)
    
    popupWindow.open()
  
   
def show_error(message):
    popupWindow = ErrorPopup()
    popupWindow.error_label.text = str(message)
    
    popupWindow.open()
    
    
class sk_wrapper():
    def __init__(self, model, scaler):
        self.model = joblib.load(model)
        self.scaler = joblib.load(scaler)
    
    def predict(self, image):
        trs = np.reshape(image, (1, 784))
        trs2 = self.scaler.transform(trs)
        
        return self.model.predict_proba(trs2)
    
    
def main():
    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
    MainApp().run()
    #combo_model = sk_wrapper("models/forest.sav", "models/scaler")
    #joblib.dump(combo_model, "ForestClassifier.sav")
    

if __name__ == "__main__":  
    main()
    