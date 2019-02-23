import os

if __name__ == "__main__":
    i = 0
    for filename in os.listdir("D:\\Documents\\csv_ll"):
        if len(filename)>30:
            target = filename[:23] + ".csv"
            os.rename("D:\\Documents\\csv_ll\\"+filename, "D:\\Documents\\csv_chb_ll_lstmpredicted\\"+target)
