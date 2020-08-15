from training import preprocessing as prc
import joblib
import numpy as np
from kivy.core.window import Window

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.config import Config

import tkinter as tk
from tkinter import filedialog

from kivy.core.image import Image as CoreImage
from PIL import Image
from io import BytesIO


class WindowManager(ScreenManager):
    pass


class LineRectangle(Widget):
    pass


class MainWindow(Screen):
    pass


class SecondWindow(Screen):
    pass


class SecondPanel(Widget):
    pass


class MainPanel(Widget):
    
    image = ObjectProperty(None)
    drawing_field = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super(MainPanel, self).__init__(**kwargs)
        self.previous_image = None
        self.pil_image = None
        
        self.move_speed = 5
        self.size_increase = 5
        
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down, on_key_up=self._on_keyboard_up)
         
    def show_file_dialog(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select a file", filetypes=(("Images", ".jpg .jpeg .png"), ("All files", "*.*")))
        
        return file_path
         
    def get_image_position(self):
        X_POS = self.ids.image.center_x - self.ids.image.norm_image_size[0] / 2
        Y_POS = self.ids.image.center_y - self.ids.image.norm_image_size[1] / 2
        
        return (X_POS, Y_POS)
              
    def crop(self):
        if(self.pil_image is None):
            return     
        
        self.previous_image = self.pil_image
    
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
        
        selection_x, selection_y = self.get_image_position()
        self.drawing_field.selection.pos = (selection_x - 10, selection_y - 10)
        self.drawing_field.selection.size[0] = self.ids.image.norm_image_size[0] + 20
        self.drawing_field.selection.size[1] = self.ids.image.norm_image_size[1] + 20
           
    def display_pil_image(self):
        data = BytesIO()
        self.pil_image.save(data, format='png')
        data.seek(0) 
        im = CoreImage(BytesIO(data.read()), ext='png')
        self.image.texture = im.texture 
        
    def load(self):
        file_name = (self.show_file_dialog())
        
        if(file_name == () or file_name == ""):
            return
        
        self.pil_image = Image.open(file_name)
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
            if(self.previous_image is not None):
                self.pil_image = self.previous_image
                self.display_pil_image()
        elif keycode[1] == '1':
            self.pil_image = Image.open("photos/1.jpg")
            self.display_pil_image()
        elif keycode[1] == '2':
            self.pil_image = Image.open("photos/2.jpg")
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
        
         
class DrawingField(Widget):
    
    selection = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super(DrawingField, self).__init__(**kwargs)
        
        self.move_speed = 5
        self.size_increase = 5

    def on_touch_down(self, touch):
         
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
        
        self.selection.pos = touch.pos
        self.selection.pos[0] -= self.selection.size[0] / 2
        self.selection.pos[1] -= self.selection.size[1] / 2


class MainApp(App):
    def build(self):
        kv = Builder.load_file("my.kv")
        self.title = 'Hadwritten digit recognition'
        return kv


def clear_image(file_name, corner, box_side):
    image = Image.open(file_name).convert('L')

    box = (corner[0], corner[1], corner[0] + box_side, corner[1] + box_side)   
    small = image.crop(box).resize((28, 28))
    np_data = np.invert(np.reshape(small, (1, 784)))
    clean = np.where(np_data < 110, 0, np_data)
    
    prc.draw_digit(clean)
    
    return clean


def test():
    file_name = 'photos/1.jpg'
    corner = (105, 180)
    side = 120
    
    digit = clear_image(file_name, corner, side)
    
    scaler = joblib.load("models/scaler")
    model = joblib.load("models/forest.sav")
    
    transformed = scaler.transform(digit)
    print(model.predict(transformed))
    
    
def main():
    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
    MainApp().run()
    

if __name__ == "__main__":
    main()
    