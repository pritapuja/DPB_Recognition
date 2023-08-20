import time
import cv2
from ImageProcessing import FrameProcessor
from tkinter import *
import tkinter.font as font
from PIL import ImageTk, Image
from tkinter import filedialog

window_name = 'Citra Tasbih Digital'
version = '_7_2'  # Versi pelatihan saat ini

erode = 2
threshold = 37
adjustment = 11
iterations = 3
blur = 3

std_height = 90
frameProcessor = FrameProcessor(std_height, version, True)

start_y = 20
current_stack_y = 0
current_stack_x = 0
current_stack_width = 0
min_width = 200
max_stack_y = 450
y_buffer = 25
x_buffer = 10

class GUI:
    def __init__(self, master):
        master.title("Aplikasi Pengenalan Angka pada Citra Tasbih Digital")
        master.minsize(410, 520)
        master.iconbitmap('C:/Users/user x/PycharmProjects/CTD_TA_Prita/img/python2.ico')

        # inisialisasi nilai
        self.hidden0 = 0
        self.hidden1 = 0
        self.hidden2 = 0
        self.btnOpenPressed_1 = 0
        self.btnOpenPressed_2 = 0
        self.btnProcessPressed_1 = 0
        self.btnProcessPressed_2 = 0

        # atur font yang dapat digunakan kembali
        self.font1 = font.Font(family='Comic Sans MS', size=16)
        self.font2 = font.Font(family='Comic Sans MS', size=14)
        self.font3 = font.Font(family='Comic Sans MS', size=12)
        self.font4 = font.Font(family='Sans Serif', size=10)
        self.font5 = font.Font(family='Comic Sans MS', size=13)
        self.font6 = font.Font(family='Comic Sans MS', size=12, underline='True')
        self.font7 = font.Font(family='Comic Sans MS', size=10)

        self.frame1(master)

    def createFrame(self, _frame):
        frame = Frame(_frame)  # buat frame yang dapat digunakan kembali
        return frame


    def frame1(self, master):
        global img_frame
        global icon_open
        global icon_process2
        global icon_next_3
        global icon_clear4
        global canvas_tasbih
        global canvas

        global btnClear_img

        self.frame_1 = self.createFrame(master)

        mainTitle = Label(self.frame_1, text="Pengenalan Angka pada", font=self.font5)
        mainTitle1 = Label(self.frame_1, text="Citra Tasbih Digital", font=self.font5)

        mainTitle.pack(pady=(15, 0))
        mainTitle1.pack(pady=(0, 15))

        img_frame = self.createFrame(self.frame_1)

        canvas_width = 200
        canvas_height = 250

        canvas = Canvas(img_frame,
                        width=canvas_width,
                        height=canvas_height, border=3)
        canvas.grid(row=0, column=0, columnspan=3, pady=(0, 15))

        canvas_tasbih = ImageTk.PhotoImage(file='img/50000.jpeg')
        canvas.create_image(6, 6, anchor=NW, image=canvas_tasbih)


        icon1 = Image.open('img/folder1.png')
        icon1 = icon1.resize((25, 25))
        icon_open = ImageTk.PhotoImage(icon1)
        btnOpen = Button(img_frame, text="Buka", font=self.font3, image=icon_open,
                           compound=LEFT, padx=8, relief=RAISED, bd=1)
        btnOpen.bind("<Button-1>", self.openImg)
        btnOpen.grid(row=1, column=0, pady=10, padx=10)

        icon2 = Image.open('img/process.png')
        icon2 = icon2.resize((25, 25))
        icon_process2 = ImageTk.PhotoImage(icon2)
        btnProcess_img = Button(img_frame, text="Pengenalan", font=self.font3, image=icon_process2,
                                compound=LEFT, padx=8, relief=RAISED, bd=1)
        btnProcess_img.bind("<Button-1>", self.openProcessImg)
        btnProcess_img.grid(row=1, column=1, pady=10, padx=10)


        myLabel1 = Label(img_frame, text="Hasil:", font=self.font3)
        myLabel1.grid(row=2, column=0, pady=(0, 20), padx=10)

        icon3 = Image.open('img/nextt.png')
        icon3 = icon3.resize((25, 25))
        icon_next_3 = ImageTk.PhotoImage(icon3)
        btnNext_img = Button(img_frame, text="Lanjut", font=self.font3, image=icon_next_3,
                                compound=RIGHT, padx=8, relief=RAISED, bd=1)
        btnNext_img.bind("<Button-1>", self.hide_frame_0)
        btnNext_img.grid(row=1, column=2, pady=10, padx=10)
        # btnNext_img.grid(row=2, column=2, pady=10, padx=10)

        label_lanjut = Label(img_frame, text="Note: Tekan tombol 'Lanjut' untuk", font=self.font4)
        label_lanjut.grid(row=3, column=0, columnspan=3, pady=(15, 0))

        label_lanjut1 = Label(img_frame, text="mengenali citra tasbih digital LCD", font=self.font4)
        label_lanjut1.grid(row=4, column=0, columnspan=3, pady=(0, 20), padx=10)

        img_frame.pack()
        self.frame_1.pack(fill=BOTH)

    def frame2(self, master):
        global img_frame
        global icon_open
        global icon_process2
        global icon_clear3
        global icon_kembali_4
        global canvas_tasbih2
        global canvas2
        global btnClear_img

        self.frame_2 = self.createFrame(master)

        mainTitle = Label(self.frame_2, text="Pengenalan Angka pada", font=self.font5)
        mainTitle1 = Label(self.frame_2, text="Citra Tasbih Digital", font=self.font5)
        mainTitle2 = Label(self.frame_2, text="[LCD]", font=self.font5)

        mainTitle.pack(pady=(15, 0))
        mainTitle1.pack()
        mainTitle2.pack(pady=(0, 15))

        img_frame = self.createFrame(self.frame_2)

        canvas_width = 200
        canvas_height = 100

        canvas2 = Canvas(img_frame,
                        width=canvas_width,
                        height=canvas_height, border=3)
        canvas2.grid(row=0, column=0, columnspan=3, pady=(0, 15))

        canvas_tasbih2 = ImageTk.PhotoImage(file='img/3000.jpg')
        canvas2.create_image(6, 6, anchor=NW, image=canvas_tasbih2)

        icon1 = Image.open('img/folder1.png')
        icon1 = icon1.resize((25, 25))
        icon_open = ImageTk.PhotoImage(icon1)
        btnOpen = Button(img_frame, text="Buka", font=self.font3,
                         image=icon_open, compound=LEFT, padx=8, relief=RAISED, bd=1)
        btnOpen.bind("<Button-1>", self.openImg_manual)
        btnOpen.grid(row=1, column=0, pady=10, padx=10)

        icon2 = Image.open('img/process.png')
        icon2 = icon2.resize((25, 25))
        icon_process2 = ImageTk.PhotoImage(icon2)
        btnProcess_img = Button(img_frame, text="Pengenalan", font=self.font3, image=icon_process2,
                                compound=LEFT, padx=8, relief=RAISED, bd=1)
        btnProcess_img.bind("<Button-1>", self.openProcessImgManual)
        btnProcess_img.grid(row=1, column=1, pady=10, padx=10)


        myLabel1 = Label(img_frame, text="Hasil:", font=self.font3)
        myLabel1.grid(row=2, column=0, pady=(0, 20), padx=10)

        icon_4 = Image.open('img/back.png')
        icon4 = icon_4.resize((25, 25))
        icon_kembali_4 = ImageTk.PhotoImage(icon4)
        btnBack_img = Button(img_frame, text="Kembali", font=self.font3,
                                image=icon_kembali_4,
                                compound=RIGHT, padx=8, relief=RAISED, bd=1)
        btnBack_img.bind("<Button-1>", self.hide_frame_0)
        btnBack_img.grid(row=1, column=2, pady=10, padx=10)

        img_frame.pack()
        self.frame_2.pack(fill=BOTH)

    def hide_frame_0(self, event):
        # Tujuan frame
        if self.hidden0 == 0:
            self.frame_1.destroy()
            self.hidden0 = 1
            if self.hidden0 == 1:
                self.hidden1 = 0
                self.frame2(root)
                print("Navigate to page 2")

        elif self.hidden1 == 0:
            self.frame_2.destroy()
            self.hidden1 = 1
            if self.hidden1 == 1:
                self.hidden0 = 0
                self.frame1(root)
                print("Navigate to page 1")

    def openImg(self, event):
        try:
            if self.btnOpenPressed_1 == 0:
                self.btnOpenPressed_1 = 1
            else:
                self.btnOpenPressed_1 = 2

            if self.btnOpenPressed_1 is 2 and self.filename is not '':
                print('img destroyed')
                myLabel4.destroy()


            self.filename = filedialog.askopenfilename(initialdir="C:/Users/user x/PycharmProjects/CTD_TA_Prita/tests/tasbih_asli", title="select a file",
                                              filetypes=(("jpeg files", "*.jpeg"), ("all files", "*.*")))
            print(self.filename)
            self.displayOpenImg(self.filename)
        except:

            self.filename = filedialog.askopenfilename(
                initialdir="C:/Users/user x/PycharmProjects/CTD_TA_Prita/tests/tasbih_asli", title="select a file",
                filetypes=(("jpeg files", "*.jpeg"), ("all files", "*.*")))
            print(self.filename)
            self.displayOpenImg(self.filename)

    def openImg_manual(self, event):
        try:
            if self.btnOpenPressed_2 == 0:
                self.btnOpenPressed_2 = 1
            else:
                self.btnOpenPressed_2 = 2

            if self.btnOpenPressed_2 is 2 and self.filename is not '':
                print('img destroyed')
                labelm8.destroy()


            self.filename = filedialog.askopenfilename(initialdir="C:/Users/user x/PycharmProjects/CTD_TA_Prita/tests/tasbih_lcd", title="select a file",
                                              filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")))
            print(self.filename)
            self.displayOpenImg_manual(self.filename)
        except:

            self.filename = filedialog.askopenfilename(
                initialdir="C:/Users/user x/PycharmProjects/CTD_TA_Prita/tests/tasbih_lcd", title="select a file",
                filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")))
            print(self.filename)
            self.displayOpenImg_manual(self.filename)

    def displayOpenImg(self, path):
        try:
            # Untuk menampilkan gambar yang dipilih
            global img_open
            global show_img_open

            canvas.destroy()
            selectedImg = Image.open(path)
            self.w, self.h = selectedImg.size
            self.imgSize()  # call this function to adjust the image size for display

            resizedImg = selectedImg.resize(self.size)
            img_open = ImageTk.PhotoImage(resizedImg)
            show_img_open = Label(img_frame, image=img_open)
            show_img_open.grid(row=0, column=0, columnspan=3, pady=(0, 15))
        except:
            selectedImg = Image.open(path)
            self.w, self.h = selectedImg.size
            self.imgSize()  # call this function to adjust the image size for display

            resizedImg = selectedImg.resize(self.size)
            img_open = ImageTk.PhotoImage(resizedImg)
            show_img_open = Label(img_frame, image=img_open)
            show_img_open.grid(row=0, column=0, columnspan=3, pady=(0, 15))

    def displayOpenImg_manual(self, path):
        try:
            # Untuk menampilkan gambar yang dipilih
            global img_open_manual
            global show_img_open_manual

            canvas2.destroy()
            selectedImg = Image.open(path)
            self.w, self.h = selectedImg.size
            self.imgSize()  # call this function to adjust the image size for display

            resizedImg = selectedImg.resize(self.size)
            img_open_manual = ImageTk.PhotoImage(resizedImg)
            show_img_open_manual = Label(img_frame, image=img_open_manual)
            show_img_open_manual.grid(row=0, column=0, columnspan=3, pady=(0, 15))
        except:
            selectedImg = Image.open(path)
            self.w, self.h = selectedImg.size
            self.imgSize()  # call this function to adjust the image size for display

            resizedImg = selectedImg.resize(self.size)
            img_open_manual = ImageTk.PhotoImage(resizedImg)
            show_img_open_manual = Label(img_frame, image=img_open_manual)
            show_img_open_manual.grid(row=0, column=0, columnspan=3, pady=(0, 15))

    def imgSize(self):
        # Untuk menyesuaikan ukuran gambar yang ditampilkan

        if self.h <= 30:
            self.size = (int(self.w * 2), int(self.h * 2))
        elif 31 <= self.h <= 500:
            self.size = (int(self.w * 0.6), int(self.h * 0.6))
        elif 501 <= self.h <= 1000:
            self.size = (int(self.w * 0.4), int(self.h * 0.4))
        elif 1001 <= self.h <= 2300:
            self.size = (int(self.w * 0.2), int(self.h * 0.2))
        else:
            self.size = (int(self.w * 0.1), int(self.h * 0.1))


    def openProcessImg(self, event):
        try:

            if self.btnProcessPressed_1 == 0:
                self.btnProcessPressed_1 = 1
            else:
                self.btnProcessPressed_1 = 2

            if self.btnProcessPressed_1 is 2:
                myLabel4.destroy()


            img_file = self.filename

            frameProcessor.set_image(img_file)
            self.processImage()
        except:
            print('gambar belum di input')


    def openProcessImgManual(self, event):
        try:
            if self.btnProcessPressed_2 == 0:
                self.btnProcessPressed_2 = 1
            elif self.btnProcessPressed_2 == 2:
                labelm8.destroy()

            img_file = self.filename

            frameProcessor.set_image(img_file)
            self.process_image_manual()
        except:
            print('gambar belum di input')

    def show_img(self, name, img):
        global current_stack_y, current_stack_x, current_stack_width
        height, width = img.shape[:2]  # untuk mengembalikan tinggi & lebar matrix gambar
        if width < min_width:
            width = min_width
        if width > current_stack_width:
            current_stack_width = width
        cv2.imshow(name, img)
        cv2.moveWindow(name, current_stack_x, current_stack_y)
        current_stack_y += height + y_buffer
        if current_stack_y > max_stack_y:
            current_stack_y = start_y
            current_stack_x += current_stack_width + x_buffer

    def reset_tiles(self):
        global current_stack_x, current_stack_y, current_stack_width
        current_stack_x = 0
        current_stack_y = 0
        current_stack_width = 0

    def processImage(self):

        global myLabel4

        self.reset_tiles()
        start_time = time.time()
        debug_images, output = frameProcessor.process_image(blur, threshold, adjustment, erode, iterations)

        for image in debug_images:
            self.show_img(image[0], image[1])

        print("Processed image in %s seconds" % (time.time() - start_time))
        cv2.imshow(window_name, frameProcessor.drawROI)
        cv2.moveWindow(window_name, 600, 600)

        myLabel4 = Label(img_frame, text=output, font=self.font1)
        myLabel4.grid(row=2, column=1, pady=(0, 20), padx=20)


    def process_image_manual(self):
        global labelm8

        self.reset_tiles()
        start_time = time.time()
        debug_images, output = frameProcessor.process_image_manual(blur, threshold, adjustment, erode, iterations)

        for image in debug_images:
            self.show_img(image[0], image[1])

        print("Processed image in %s seconds" % (time.time() - start_time))
        cv2.imshow(window_name, frameProcessor.img)

        labelm8 = Label(img_frame, text=output, font=self.font1)
        labelm8.grid(row=2, column=1, pady=(0, 20), padx=20)

root = Tk()
gui = GUI(root)
root.mainloop()


