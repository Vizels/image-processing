import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import cv2 as cv
import io


M = 64
m = -M/2
propeller_num = 3
animspeed = 50
ss = 16 #Shutter Size
frames_number = int(256/ss)


fig, axs= plt.subplots(ncols=2,subplot_kw={'projection':'polar'})

#decrease whitespace
plt.tight_layout(pad=0, h_pad=None, w_pad=None, rect=None)

#hide axis measures
axs[0].set_xticklabels([])
axs[0].set_yticklabels([])

#hide second plot
axs[1].set_xticklabels([])
axs[1].set_yticklabels([])
axs[1].grid(0)
axs[1].axis('off')

#set x and y
x = np.linspace(0, 2*np.pi, 1000)
line, = axs[0].plot(x, np.sin(propeller_num*x + (m*np.pi)/10))

#this subplot will show aliased image
axs[1] = fig.add_subplot(1, 2, 2)#, projection='rectilinear')
axs[1].axis('off')
axs[1].grid(0)

#save image to buffer to get shapes
temp = io.BytesIO()
plt.savefig(temp, format='png', bbox_inches='tight', pad_inches=0, dpi = 100) #53.5 for 256 pixels height, 99.3 after crop
temp.seek(0)
temp_img = cv.imdecode(np.frombuffer(temp.getvalue(), np.uint8), cv.IMREAD_COLOR)
temp.close()

(height, width) = temp_img.shape[:2]

#empty array for image
frames_array = np.zeros((256, int(width/2), 3), dtype=int)



def save_frame(i): #function to get frames from animations
    #image to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi = 100)
    buf.seek(0)

    #image to var
    img = cv.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv.IMREAD_COLOR)
    buf.close()

    #get every l lines from every animation shot
    frames_array[(i*ss):(i*ss)+ss,:,:] = img[(i*ss):(i*ss)+ss, 0:int(width/2), : ]


def animate(m, line):
    line.set_ydata(np.sin(propeller_num*x + ((m+1)*np.pi)/10)) 
    save_frame(int(m))
    axs[1].clear()
    axs[1].imshow(frames_array)
    return line,

ani = anim.FuncAnimation(
    fig, animate, frames = frames_number, fargs=[line], interval=animspeed, blit=False, repeat = False)

plt.show()