import keras
import matplotlib.pyplot as plt
import numpy as np
import serial
import tkinter as tk
import threading
import cv2
import tensorflow as tf
from tensorflow.python.keras.saving.saved_model.load import input_layer

ser = serial.Serial(port='COM5', baudrate=115200, timeout=2)
photo_taken = False  # 사진 촬영 여부 확인 변수

# 메인 Tkinter 윈도우 (UI 요소 제거)
window = tk.Tk()
window.withdraw()  # 기본 창 숨기기 (팝업만 띄우도록 설정)


def capture_photo():
    # 카메라로 사진을 찍고 저장하는 함수
    global photo_taken
    file_name = "captured_photo.jpg"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return None

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(file_name, frame)
        print(f"사진이 저장되었습니다: {file_name}")
        photo_taken = True  # 사진이 찍혔다고 표시
    #    ser.close()  # 아두이노 종료
        window.quit()  # Tkinter 루프 종료 (중요!)
    else:
        print("사진을 캡처할 수 없습니다.")

    cap.release()
    cv2.destroyAllWindows()
    return file_name



def show_popup():
    # PIR 센서 감지 시 사진 촬영 여부를 묻는 팝업을 띄움
    popup = tk.Toplevel()
    popup.title("사진 촬영")
    popup.geometry("250x150")

    label = tk.Label(popup, text="사진을 찍으시겠습니까?", font=("Arial", 12))
    label.pack(pady=10)

    btn_take_photo = tk.Button(popup, text="촬영", command=lambda: [capture_photo(), popup.destroy()])
    btn_take_photo.pack(pady=5)

    btn_cancel = tk.Button(popup, text="취소", command=popup.destroy)
    btn_cancel.pack(pady=5)


def serial_read():
    # 시리얼 데이터 읽기: PIR 센서 감지 시 팝업 띄우기
    global photo_taken
    while not photo_taken:  # 사진이 안 찍힌 경우만 계속 감지
        try:
            data = ser.readline().decode('utf-8').strip()
            if data:
                print(f"Received data: {data}")

                if data == "TAKE_PHOTO":
                    window.after(0, show_popup)  # 감지되면 팝업 띄우기

        except Exception as e:
            print(f"Serial read error {e}")
            break


# 시리얼 읽기 스레드 시작
thread = threading.Thread(target=serial_read, daemon=True)
thread.start()

window.mainloop()  # → 사진 촬영 후 quit() 호출되면 종료됨

# 아두이노가 종료된 후 실행될 코드
if photo_taken:
    print("아두이노 종료됨, 이후 코드 실행 시작...")
    face_images = []  # 얼굴 이미지 리스트
    for i in range(100):  # 20개의 얼굴 사진 불러오기
        file = './myfaces/' + 'img{0:02d}.jpg'.format(i + 1)
        img = cv2.imread(file)
        img = cv2.resize(img, dsize=(64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_images.append(img)


    def show_image(row: int, col: int, image: list[np.ndarray]) -> None:
        fig, ax = plt.subplots(row, col, figsize=(col * 2, row * 2))  # 크기 조정

        # 1x1일 경우 단일 축이므로 리스트로 변환
        if row == 1 and col == 1:
            ax = [ax]

        for i in range(row):
            for j in range(col):
                index = i * col + j
                if index >= len(image):  # 이미지 개수를 초과하는 경우 방지
                    break

                if row == 1:
                    axis = ax[j]  # 1차원 배열이므로 j 인덱싱
                elif col == 1:
                    axis = ax[i]  # 1차원 배열이므로 i 인덱싱
                else:
                    axis = ax[i, j]  # 2차원 배열 인덱싱

                axis.get_xaxis().set_visible(False)
                axis.get_yaxis().set_visible(False)
                axis.imshow(image[index])

        plt.show()

        return None

    show_image(row=10, col=10, image=face_images)

strange_images = []
for i in range(55):
    file = './strange/' + 'img{0:02d}.jpg'.format(i + 1)
    img = cv2.imread(file)
    img = cv2.resize(img, dsize=(64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    strange_images.append(img)

show_image(row=11, col=5, image=strange_images)

# # labes 만들기 나[1, 0] 외 [0, 1] 형태 -> 핫인코딩
# y = [(1, 0)] * len(face_images) + [(0, 1)] * len(strange_images)
# print(y)
# y = np.array(y)
# print(y)
# X = face_images + strange_images
# print(X)
# X = np.array(X)
# print(X)
#
#
# #CNN
# X = X / 255.0
#
# X_train = tf.constant(X, name='X_train')
# y_train = tf.constant(y, name='y_train')
#
# model = keras.Sequential([], name='MYFACEDETECTOR')
# input_layer = keras.Input(shape=(64, 64, 3), name='Input_layer')
# model.add(input_layer)
# model.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
# model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=2))
# model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
# model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
# model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=2))
# model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
# model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
# model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=2))
# #DNN
# model.add(keras.layers.Flatten(name='Flatten'))
# model.add(keras.layers.Dense(units=128, activation='relu', name='HD1'))
# model.add(keras.layers.Dense(units=32, activation='relu', name='HD2'))
# model.add(keras.layers.Dense(units=2, activation='softmax', name='Output_Layer'))
# model.summary()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(x=X, y=y, epochs=300)
# print(f'훈련 정확도 : {model.evaluate(x=X, y=y)}')
# save = model.save('FACE_Detector.keras')

test_images = []
file = 'captured_photo.jpg'
img = cv2.imread(file)
img = cv2.resize(img, dsize=(64, 64))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
test_images.append(img)
test_images = tf.constant(test_images)
print(test_images.shape)

cnn_model = keras.models.load_model("FACE_Detector.keras")
predict = cnn_model.predict(test_images)
print(predict)
show_image(row=1, col=1, image=test_images)

if np.argmax(predict) == 0:  # [[1. 0.]]이면 0번 인덱스가 가장 큼
    ser.write(b'ON\n')  # 'ON'을 바이트 문자열로 변환하여 전송
    print("Sent: ON")
else:  # [[0. 1.]]이면 1번 인덱스가 가장 큼
    ser.write(b'OFF\n')
    print("Sent: OFF")


