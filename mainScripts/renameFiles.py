import os
import re

if __name__ == "__main__":
    i = 0
    for filename in os.listdir("D:\\Documents\\csv_fft"):
        index = filename.find("l")
        if (index > 0):
            target = filename[:index] + "csv"
            os.rename("D:\\Documents\\csv_fft\\"+filename, "D:\\Documents\\csv_chb_fft_lstmpredicted\\"+target)
