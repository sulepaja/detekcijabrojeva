from lineDetection import *
import cv2
import numpy as numpy
import matplotlib as plt
from skimage import color
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import *
from scipy import ndimage
from sklearn.datasets import fetch_mldata
from vector import *
import matplotlib.pyplot as plt

#LOADED VIDEO
video = "videos/video-1.avi"
v = cv2.VideoCapture(video)

#STARTING AND END POINT OF TWO LINES
lower=[]
upper=[]

#ORIGINAL MNIST DATASET AND OUR 28x28 MNIST DATASET
mnist = fetch_mldata('MNIST original')
mnistNumbers = []

ret, frame = v.read()

if ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    cv2.imwrite('foundLines.jpg', gray)
    cv2.waitKey()
# When everything done, release the capture
v.release()
cv2.destroyAllWindows()

lower, upper = prepareHough(frame,gray,video)
lowerLine = [(lower[0][0], lower[0][1]), (lower[0][2],lower[0][3])]
upperLine = [(upper[0][0], upper[0][1]) ,(upper[0][2], upper[0][3])]

id = -1
final_sum = 0

def putInLeftCorner(img):

    minx = 2000
    miny = 2000
    maxx = -1
    maxy = -1
    newImage = numpy.zeros((28,28),numpy.uint8)

    try:
        # LABEL IMAGE REGIONS
        label_img = label(img)
        regions = regionprops(label_img)
        # print("Number of regions is: ", len(regions))

        # plt.imshow(img, 'gray')
        # plt.show()
        # cv2.waitKey()

        for region in regions:
            bbox = region.bbox
            if bbox[0]<minx:
                minx=bbox[0]
            if bbox[1]<miny:
                miny=bbox[1]
            if bbox[2]>maxx:
                maxx=bbox[2]
            if bbox[3]>maxy:
                maxy=bbox[3]

        numberWidth = maxx-minx
        numberHeight = maxy-miny

        newImage[0:numberWidth, 0:numberHeight] = newImage[0:numberWidth, 0:numberHeight] + img[minx : maxx, miny : maxy]
        return newImage

    except ValueError:
        print("An error has been caught")
        pass

def additionalDilate2(img, video):
    if video == "videos/video-5.avi":
        kernel = numpy.ones((2, 2), numpy.uint8)
        img = cv2.dilate(img, kernel)
    return img

def additionalDilate3(img0,video):
    if video != "videos/video-6.avi":
        kernel = numpy.ones((3,3),numpy.uint8)
        img0 = cv2.dilate(img0, kernel)
    return img0

def reshapeMNIST(mnist):
    for i in range(len(mnist.data)):
        image = mnist.data[i].reshape(28,28)  #MNIST.DATA ARE ORIGINAL MNIST IMAGES
        bin_img = ((color.rgb2gray(image) / 255.0) >= 0.88).astype('uint8')
        bin_img = putInLeftCorner(bin_img)
        mnistNumbers.append(bin_img) #RESIZED LIST

def isInRange(dist, num, numbers):
    res=[]
    num_center = num['center']
    for n in numbers:
        it_center = n['center']
        if(distance(num_center, it_center) < dist):
            res.append(n)
    return res

def nextId():
    global id
    id = id+ 1
    return id

def findNumber(image):
    minSum = 9999
    ret = -1

    for i in range (len(mnistNumbers)):
        sum = 0
        mnist_img = mnistNumbers[i]
        sum = numpy.sum(mnist_img != image)
        if sum < minSum:
            minSum = sum
            ret = mnist.target[i]
    return ret

def changeImage(image):
    img_temp = color.rgb2gray(image) >= 0.88
    img_temp = (img_temp).astype('uint8')
    new_image = putInLeftCorner(img_temp)
    new_image = additionalDilate2(new_image,video) #TURN ON: video-5
    rez = findNumber(new_image)

    return rez

def findClosest(list, elem):
    min = list[0]
    for obj in list:
        if distance(obj['center'], elem['center']) < distance(min['center'], elem['center']):
            min = obj
    return min

def main():
    kernel = numpy.ones((2,2), numpy.uint8)
    #v=cv2.VideoCapture(video)
    v.open(video)
    f=0
    numbers=[]
    allowedDistance = 20

    print ("Starts resizing MNIST data. ")
    reshapeMNIST(mnist)
    print ("Finished resizing MNIST data. ")

    print ("Starts with frame iteration")
    while 1:

        ret, currentFrame = v.read()
        if not ret:
            break

        #DEFINE THRESHOLD LOWER AND UPPER (BGR)
        threshold_lower = numpy.array([225, 225, 225], dtype="uint8") #ALMOST WHITE
        threshhold_upper = numpy.array([255,255,255], dtype="uint8") #WHITE

        #REMOVE ALL COLOURS THAT ARE NOT BETWEEN TWO BOUNDARIES
        mask = cv2.inRange(currentFrame, threshold_lower, threshhold_upper)

        #cv2.imshow('currentFrame', mask)
        #cv2.waitKey()

        img = mask * 1.0
        imgCopy = mask * 1.0

        img = cv2.dilate(img,kernel)
        img = cv2.dilate(img, kernel)
        img = additionalDilate3(img,video)  #TURN OFF FOR: video-6

        labeled, nr_objects = ndimage.label(img)
        elements = ndimage.find_objects(labeled)

        for i in range(len(elements)):
            loc = elements[i]

            #CENTER OF OBJECT
            cen = []
            cen.append((loc[1].stop + loc[1].start) /2 )
            cen.append((loc[0].stop + loc[0].start)/2)

            #DIMENSIONS
            dim = []
            dim.append(loc[1].stop - loc[1].start)
            dim.append(loc[0].stop - loc[0].start)

            if dim[0] > 11 or dim[1] > 11:
                el = {'center' : cen, 'dimension' : dim, 'frame':f}

                res = isInRange(allowedDistance,el,numbers)


                if len(res) == 0:
                    el['id'] = nextId()
                    el['hasPassed1'] = False
                    el['hasPassed2'] = False

                    if video =='videos/video-3.avi' or video == 'videos/video-4.avi'or video =='videos/video-5.avi' or video == 'videos/video-7.avi':
                        x1 = cen[0] - 10
                        y1 = cen[1] - 10
                        x2 = cen[0] + 10
                        y2 = cen[1] + 10

                    else:
                        x1 = cen[0]-14
                        y1 = cen[1]-14
                        x2 = cen[0]+14
                        y2 = cen[1]+14
                    el['value'] = changeImage(imgCopy[y1:y2, x1:x2])
                    el['image'] = imgCopy[y1:y2, x1:x2]

                    numbers.append(el)
                else:
                    elem = findClosest(res, el)
                    elem['center'] = el['center']
                    elem['frame'] = el['frame']
        for elem in numbers :
            x = f - elem['frame']
            global final_sum
            if  x < 3:
                #print(upperLine[0],":::",upperLine[1])
                dist, nearest, r = pnt2line2(elem['center'], upperLine[0], upperLine[1])
                if r>0:
                    if dist < 9 :
                        if elem['hasPassed1'] == False:
                            elem['hasPassed1'] = True
                            print ("ADDING: ") + format(elem['value'])                            
                            cv2.waitKey()
                            final_sum = final_sum + elem['value']
                            print("CURRENT SUM IS: ", format(final_sum))
                dist2, nearest2, r2 = pnt2line2(elem['center'], lowerLine[0], lowerLine[1])
                if r2 > 0:
                    if dist2 < 9:
                        if elem['hasPassed2'] == False:
                            elem['hasPassed2'] = True
                            print ("SUBTRACTING: ") + format(elem['value'])                            
                            cv2.waitKey()
                            final_sum = final_sum - elem['value']
                            print("CURRENT SUM IS: ", format(final_sum))
        f=f+1  #NEXT FRAME
        cv2.imshow('CurrentFrame', currentFrame)
        #print("FRAME NUMBER: ", f)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    print ("FINAL SUM IS: ") + format(final_sum)
    v.release()
    cv2.destroyAllWindows()


main()

#video-1 100% 14

#vodeo-3 10
#video-4 10
#vudeo-5 10
#video-7 10



