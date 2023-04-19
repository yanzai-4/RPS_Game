# Import the required libraries :
import time
from asyncio import wait
from tkinter import *
import tkinter
import tkinter.messagebox
from PIL import Image, ImageTk
import random
import cv2
import numpy as np
import copy
import simpleaudio as sa
import matplotlib
matplotlib.use('Agg')
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#Load H5 CNN Model
MODEL_PATH = os.getcwd()+'\\model\\rock_paper_scissors_cnn.h5'
model = load_model(MODEL_PATH)
model.summary()

class GUI:
    def __init__(self, window):
        self.BGM = sa.WaveObject.from_wave_file("mic/BGM.wav")
        self.main_photo = PhotoImage(file='pic/main.png')
        self.num_one_photo = PhotoImage(file='pic/num_one.png')
        self.num_two_photo = PhotoImage(file='pic/num_two.png')
        self.num_three_photo = PhotoImage(file='pic/num_three.png')
        self.empty_photo = PhotoImage(file='pic/syb_empty.png')
        self.wallpaper_photo = PhotoImage(file='pic/wallpaper.png')
        self.unknown_photo = Image.open("pic/syb_unknown.png")
        self.rock_photo = Image.open("pic/rock.png")
        self.paper_photo = Image.open("pic/paper.png")
        self.scissor_photo = Image.open("pic/scissor.png")
        self.scissor_image = ImageTk.PhotoImage(self.scissor_photo)
        self.paper_image = ImageTk.PhotoImage(self.paper_photo)
        self.rock_image = ImageTk.PhotoImage(self.rock_photo)
        self.unknown_image = ImageTk.PhotoImage(self.unknown_photo)
        self.click_sound = sa.WaveObject.from_wave_file("mic/click.wav")
        self.win_sound = sa.WaveObject.from_wave_file("mic/win.wav")
        self.lose_sound = sa.WaveObject.from_wave_file("mic/lose.wav")
        self.tie_sound = sa.WaveObject.from_wave_file("mic/tie.wav")
        self.back_button = Button(text="BACK", font=('Arial', 15), command=self.back)
        self.final_result = self.final_result = Label(text="", font=('Arial', 40), bg="#FFFFFF")
        self.reset_button = Button(text="RESET", font=('Arial', 15), command=self.reset)
        self.wallpaper = ''
        self.loading_title = ''
        self.start_page_photo = ''
        self.start_page_title = ''
        self.game_page_title = ''
        self.start_button = ''
        self.option_button = ''
        self.quit_button = ''
        self.pc_name = ''
        self.player_name = ''
        self.pc_result = ''
        self.player_result = ''
        self.pc_status = ''
        self.player_status = ''
        self.play_button = ''
        self.choice = ''
        self.timer = Label(image=self.empty_photo)
        self.window = window

    def set_window(self):
        self.BGM.play()
        self.window.title("Rock-Paper-Scissor")
        self.window.geometry('1280x720')
        self.window.iconbitmap('pic/Game.ico')
        self.window.configure(bg="#FFFFFF")
        self.window.resizable(width=False, height=False)


    def main_menu(self):
        self.wallpaper = Label(imag=self.wallpaper_photo)
        self.start_page_photo = Label(imag=self.main_photo)
        self.start_page_title = Label(text="Rock-Paper-Scissor", font=('Arial', 50), bg="#FFFFFF")
        self.start_button = Button(text="START", font=('Arial', 25), command=self.instructions)
        self.option_button = Button(text="OPTION", font=('Arial', 25), command=self.option)
        self.quit_button = Button(text="QUIT", font=('Arial', 25), command=self.quit)
        self.wallpaper.place(x=0, y=0)
        self.start_page_photo.place(x=300, y=50, width=700, height=250)
        self.start_page_title.place(x=360, y=330)
        self.start_button.place(x=580, y=450)
        self.option_button.place(x=570, y=530)
        self.quit_button.place(x=595, y=610)
        self.reset_button.destroy()
        self.back_button.destroy()
        self.final_result.destroy()

    def clean_menu(self):
        self.back_button = Button(text="BACK", font=('Arial', 15), command=self.back)
        self.reset_button = Button(text="RESET", font=('Arial', 15), command=self.reset)
        self.start_page_photo.destroy()
        self.start_page_title.destroy()
        self.start_button.destroy()
        self.option_button.destroy()
        self.quit_button.destroy()
        self.reset_button.place(x=1080, y=40)
        self.back_button.place(x=1180, y=40)

    def back(self):
        self.click_sound.play()
        self.main_menu()
        self.game_page_title.destroy()
        self.pc_name.destroy()
        self.player_name.destroy()
        self.pc_result.destroy()
        self.player_result.destroy()
        self.pc_status.destroy()
        self.player_status.destroy()
        self.final_result.destroy()
        self.timer.destroy()

    def reset(self):
        self.click_sound.play()
        self.camera()

    def quit(self):
        self.window.quit()

    def option(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        vr = volume.GetVolumeRange()
        self.Volume_label = Label(text="Volume Control", font=('Arial', 20), bg="#FFFFFF")
        print(vr)

        def sound():
            vl = self.sound_value.get()
            vl /= 1.5238
            vl -= 65.625
            if(vl >= 0):
                volume.SetMasterVolumeLevel(0, None)
            else:
                volume.SetMasterVolumeLevel(vl, None)

        def check():
            camera = cv2.VideoCapture(0)
            if(camera.isOpened()):
                tkinter.messagebox.showinfo('INFO', 'WebCam is working           \n')
            else:
                tkinter.messagebox.showwarning('ERROR', 'WebCam is not working          \n')
        self.click_sound.play()
        self.clean_menu()
        self.reset_button.destroy()
        self.sound_value = tkinter.Scale(from_=0, to=100, orient=tkinter.HORIZONTAL, tickinterval=20, length=200, bg="#FFFFFF")
        vl = volume.GetMasterVolumeLevel()
        vl += 65.625
        vl *= 1.5238
        self.sound_value.set(vl)
        self.Volume_label.place(x=545, y=200)
        self.sound_value.place(x=540, y=270)
        self.sound_button = Button(text="SET", font=('Arial', 15), command=sound)
        self.sound_button.place(x=615, y=360)
        self.sound_button = Button(text="Check WebCam Connect", font=('Arial', 15), command=check)
        self.sound_button.place(x=530, y=440)

    def instructions(self):
        self.click_sound.play()
        mes = tkinter.messagebox.askokcancel('Instructions', '1.Make sure your camera is on           \n'
                                                             '2 Make sure you are not in the camera   \n'
                                                             '3 Make sure the surroundings are bright \n')
        if mes:
            self.start()

    def start(self):
        var = StringVar()
        self.clean_menu()
        self.loading_title = Label(text="Loading...", font=('Arial', 30), bg="#FFFFFF")
        self.loading_title.place(x=560, y=330)
        self.window.update_idletasks()
        self.window.update()
        self.game_page_title = Label(text="Rock-Paper-Scissor", font=('Arial', 30), bg="#FFFFFF")
        self.pc_name = Label(text="PC", font=('Arial', 20), bg="#FFFFFF")
        self.player_name = Label(text="player", font=('Arial', 20), bg="#FFFFFF")
        self.pc_result = Canvas(bg='white', relief=GROOVE)
        self.player_result = Canvas(bg='white', relief=GROOVE)
        self.pc_result.create_image(200, 165, image=self.unknown_image)
        self.pc_status = Label(text="Statue:______", font=('Arial', 20), bg="#FFFFFF")
        self.player_status = Label(text="Statue:______", font=('Arial', 20), bg="#FFFFFF")
        self.final_result = Label(text="Show your hand in webcam", font=('Arial', 40), bg="#FFFFFF")
        self.timer = Label(bg="#FFFFFF")
        self.game_page_title.place(x=460, y=50)
        self.pc_name.place(x=300, y=120)
        self.player_name.place(x=910, y=120)
        self.pc_result.place(x=130, y=200, height=310, width=400)
        self.player_result.place(x=750, y=200, height=310, width=400)
        self.pc_status.place(x=230, y=520)
        self.player_status.place(x=850, y=520)
        self.final_result.place(x=330, y=570)
        self.timer.place(x=565, y=250)
        self.camera()


    def camera(self):

        def play(player_choice):
            pc_choice = random.choice(["Rock", "Paper", "Scissor"])
            info = "Statue:" + pc_choice
            self.pc_status.configure(text=info, font=('Arial', 20), bg="#FFFFFF")
            if pc_choice == "Rock":
                self.pc_result.create_image(200, 165, image=self.rock_image)
                print(player_choice)
                if player_choice == "Rock":
                    self.final_result.configure(text="        --------TIE--------")
                    self.tie_sound.play()
                elif player_choice == "Paper":
                    self.final_result.configure(text="        --------WIN--------")
                    self.win_sound.play()
                elif player_choice == "Scissor":
                    self.final_result.configure(text="        --------LOSE-------")
                    self.lose_sound.play()
            elif pc_choice == "Paper":
                self.pc_result.create_image(200, 165, image=self.paper_image)
                if player_choice == "Rock":
                    self.final_result.configure(text="        --------LOSE-------")
                    self.lose_sound.play()
                elif player_choice == "Paper":
                    self.final_result.configure(text="        --------TIE--------")
                    self.tie_sound.play()
                elif player_choice == "Scissor":
                    self.final_result.configure(text="        --------WIN--------")
                    self.win_sound.play()
            elif pc_choice == "Scissor":
                self.pc_result.create_image(200, 165, image=self.scissor_image)
                if player_choice == "Rock":
                    self.final_result.configure(text="        --------WIN--------")
                    self.win_sound.play()
                elif player_choice == "Paper":
                    self.final_result.configure(text="        --------LOSE-------")
                    self.lose_sound.play()
                elif player_choice == "Scissor":
                    self.final_result.configure(text="        --------TIE--------")
                    self.tie_sound.play()
            else:
                self.final_result.configure(text="        --------ERROR--------")
            self.window.update_idletasks()
            self.window.update()
            cv2.waitKey(500)

        camera = cv2.VideoCapture(0)

        def game():
            self.timer.configure(image=self.num_one_photo)
            self.window.update_idletasks()
            self.window.update()
            time.sleep(1)
            self.timer.configure(image=self.num_two_photo)
            self.window.update_idletasks()
            self.window.update()
            time.sleep(1)
            self.timer.configure(image=self.num_three_photo)
            self.window.update_idletasks()
            self.window.update()
            time.sleep(1)
            ret, frame = camera.read()
            pre_x=[]
            i = cv2.resize(frame[0:400,120:520], (150,150))
            img_PATH="C:\\Users\\crh01\\Desktop\\Programming\\Program_Python\\RPS-beta-2.0.1\\temp\\temp_size.jpg"
            cv2.imwrite(img_PATH,i)
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
            img_PATH="C:\\Users\\crh01\\Desktop\\Programming\\Program_Python\\RPS-beta-2.0.1\\temp\\temp_after.jpg"
            cv2.imwrite(img_PATH,i)
            pre_x.append(i)
            pre_x.append(i)
            pre_x = np.array(pre_x) / 255.0
            pre_y = model.predict(pre_x)
            e = np.argmax(pre_y[0])
            if(e==0):
                self.player_status.configure(text="Status:Rock", font=('Arial', 20), bg="#FFFFFF")
                play("Rock")
            elif(e==1):
                self.player_status.configure(text="Status:Paper", font=('Arial', 20), bg="#FFFFFF")
                play("paper")
            elif(e==2):
                self.player_status.configure(text="Status:Scissor", font=('Arial', 20), bg="#FFFFFF")
                play("Scissor")
            else:
                play("Not recognized")
            self.timer.configure(image=self.empty_photo)
            self.window.update_idletasks()
            self.window.update()

        self.game_start = Button(text="GAME", font=('Arial', 25), command=game)
        self.game_start.place(x=588, y=465)

        while camera.isOpened():
            self.loading_title.destroy()
            ret, frame = camera.read()
            frame = cv2.flip(frame, 1)
            cov = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cov)
            image_file = ImageTk.PhotoImage(img)
            self.player_result.create_image(200, 200, image=image_file)
            self.window.update_idletasks()
            self.window.update()
                    

def gui_start():
    init_window = Tk()
    screenwidth = init_window.winfo_screenwidth()
    screenheight = init_window.winfo_screenheight()
    init_window.geometry('%dx%d+%d+%d' % (1280, 720, (screenwidth - 1280) / 2, (screenheight - 720) / 2))
    RPS = GUI(init_window)
    RPS.set_window()
    RPS.main_menu()
    init_window.mainloop()


if __name__ == '__main__':
    gui_start()
