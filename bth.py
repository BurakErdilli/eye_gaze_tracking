import serial
import time

def command(x):
    comm.reset_input_buffer()
    comm.write(bytes(str(x), "utf-8"))
    time.sleep(2)
    comm.write(bytes("d", "utf-8"))


comm = serial.Serial('COM7',9600,timeout = 2)
comm.close()
comm.open()

command("y")


def main():
    pass




#k ilerisol, m ilerisag, a geri, d dur

if __name__ == "__main__":
    main()
