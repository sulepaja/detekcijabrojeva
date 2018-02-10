from __future__ import division
import numpy as numpy
import cv2
import math as math

def prepareHough(frame, gray, video):

    kernel = numpy.ones((2, 2), numpy.uint8)

    if video == 'videos/video-3.avi' or video=='videos/video-8.avi':
        gray = cv2.erode(gray, kernel)
    #gray = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 400
    maxLineGap = 5

    lines = cv2.HoughLinesP(edges,1,numpy.pi/180,50,minLineLength,maxLineGap)

    #a,b,c = lines.shape
    #for i in range(a):
        #cv2.line(frame, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    #cv2.imshow('result', frame)
    #cv2.waitKey()
    #lineNumber = len(lines)
    #cv2.imshow('edges', edges)

    if video=='videos/video-3.avi':
        return appendLines3(lines,frame)
    return appendLines(lines, frame)

def appendLines(lines, frame):
    lineNumber = len(lines)
    resenFrame = frame

    lowerLinePoints = []
    upperLinePoints = []

    minx=lines[0][0][0]
    miny=lines[0][0][1]
    maxx=lines[0][0][2]
    maxy=lines[0][0][3]

    distance = math.sqrt(math.pow(minx-maxx,2) + math.pow(miny-maxy,2))

    #MIN I MAX OBE LINIJE ZAJEDNO
    for i in range(lineNumber-1):
        for j in range (lineNumber):
            x1 = lines[i][0][0]
            y1 = lines[i][0][1]
            x2 = lines[j][0][2]
            y2 = lines[j][0][3]

            tempd = math.sqrt(math.pow(x2-x1,2) + math.pow(y2-y1,2))
            if tempd > distance:
                distance = tempd
                minx = x1
                miny = y1
                maxx = x2
                maxy = y2

    middle_pointx = (minx+maxx)/2
    middle_pointy = (miny+maxy)/2

    print("MINIMUM IS : ", minx,miny)
    print("MAXIMUM IS : ", maxx, maxy)
    print("MIDDLE IS : ", middle_pointx, middle_pointy)

    middle_pointx = int(middle_pointx)
    middle_pointy = int(middle_pointy)
    cv2.circle(resenFrame, (middle_pointx, middle_pointy), 2, (0, 0, 255), 3)

    points1 = []
    points2 = []

    for i in range(lineNumber):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]

        list_item = lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3]
        if y1<=middle_pointy:
            points1.append(list_item)
        if abs(y1-middle_pointy)<30:
            points1.append(list_item)
        if abs(y1-middle_pointy)<20:
            points2.append(list_item)
        if y1>=middle_pointy:
            points2.append(list_item)

    minxp1 = points1[0][0]
    minyp1 = points1[0][1]
    maxxp1 = points1[0][2]
    maxyp1 = points1[0][3]

    for i in range(len(points1)):
        x1 = points1[i][0]
        y1 = points1[i][1]
        x2 = points1[i][2]
        y2 = points1[i][3]

        if x1<minxp1:
            minxp1 = x1
            minyp1 = y1
        if x2>maxxp1:
            maxxp1 = x2
            maxyp1 = y2
    cv2.line(resenFrame,(minxp1,minyp1),(maxxp1, maxyp1), (0, 255, 255),3)

    appendItem = minxp1, minyp1, maxxp1, maxyp1
    upperLinePoints.append(appendItem)

    minxp2 = points2[0][0]
    minyp2 = points2[0][1]
    maxxp2 = points2[0][2]
    maxyp2 = points2[0][3]

    for i in range(len(points2)):
        x1 = points2[i][0]
        y1 = points2[i][1]
        x2 = points2[i][2]
        y2 = points2[i][3]

        if x1 < minxp2:
            minxp2 = x1
            minyp2 = y1
        if x2 > maxxp2:
            maxxp2 = x2
            maxyp2 = y2
    cv2.line(resenFrame, (minxp2, minyp2), (maxxp2, maxyp2), (255, 0, 255), 3)
    cv2.imshow('Result',resenFrame)
    cv2.imwrite("lineDetected.jpg", resenFrame);

    appendItem = minxp2, minyp2, maxxp2, maxyp2
    lowerLinePoints.append(appendItem)

    cv2.line(frame,(minx, miny), (maxx,maxy), (0,255,255),3)
    cv2.waitKey()
    return lowerLinePoints, upperLinePoints

def appendLines3(lines,frame):
    lineNumber = len(lines)
    resenFrame = frame

    lowerLinePoints = []
    upperLinePoints = []

    x1k = lines[0][0][0]
    y1k = lines[0][0][1]
    x2k = lines[0][0][2]
    y2k = lines[0][0][3]

    k = ((y2k - y1k) / (x2k - x1k))
    n1 = y1k - k * x1k

    x1m = lines[lineNumber - 1][0][0]
    y1m = lines[lineNumber - 1][0][1]
    x2m = lines[lineNumber - 1][0][2]
    y2m = lines[lineNumber - 1][0][3]

    m = ((y2m - y1m) / (x2m - x1m))
    n2 = y1m - k * x1m

    first_line = []
    second_line = []

    minx = lines[0][0][0]
    miny = lines[0][0][1]
    maxx = lines[0][0][2]
    maxy = lines[0][0][3]


    for i in range(lineNumber):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]

        i_k = (y2 - y1) / (x2 - x1)
        i_n = y1-i_k*x1

        list_item = lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3]
        if abs(n1-i_n)<10:
            first_line.append(list_item)
        if abs(n1-i_n)>20:
            second_line.append(list_item)

        #cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        #cv2.imshow('test',frame);
        #cv2.waitKey()
    sminx = second_line[0][0]
    sminy = second_line[0][1]
    smaxx = second_line[0][2]
    smaxy = second_line[0][3]

    first_line_lenght = len(first_line)
    second_line_lenght = len(second_line)

    for i in range(second_line_lenght):
        x1 = second_line[i][0]
        y1 = second_line[i][1]
        x2 = second_line[i][2]
        y2 = second_line[i][3]
        if x1<sminx:
            sminx = x1
            sminy = y1
        if x2>smaxx:
            smaxx=x2
            smaxy=y2
    cv2.line(frame,(sminx, sminy), (smaxx,smaxy), (0,255,255),3)
    appendItem = sminx, sminy, smaxx, smaxy
    upperLinePoints.append(appendItem)
    #cv2.imshow("linije", frame)
    #cv2.waitKey()

    ssminx = first_line[0][0]
    ssminy = first_line[0][1]
    ssmaxx = first_line[0][2]
    ssmaxy = first_line[0][3]

    for i in range(first_line_lenght):
        x1 = first_line[i][0]
        y1 = first_line[i][1]
        x2 = first_line[i][2]
        y2 = first_line[i][3]
        if x1 < ssminx:
            ssminx = x1
            ssminy = y1
        if x2 > ssmaxx:
            ssmaxx = x2
            ssmaxy = y2
    cv2.line(frame, (ssminx, ssminy), (ssmaxx, ssmaxy), (255, 0, 255), 3)
    appendItem = ssminx, ssminy, ssmaxx, ssmaxy
    lowerLinePoints.append(appendItem)
    cv2.imshow("linije", frame)
    cv2.waitKey()

    return lowerLinePoints, upperLinePoints