import cv2
import numpy as np
import os
import time


class UnfoldingLongExposure:


    def __init__(self, inputFolder, outputFile = None, thresholdKernelSize = (7,7), threshold=2):
        if inputFolder[-1] != '/':
            inputFolder += '/'
        self.folder = inputFolder
        self.output = outputFile
        self.thresholdKernelSize = thresholdKernelSize
        self.threshold = threshold
        self.deexposureRatio = None


    def get_treshold_binary(self, image):
        # convert the image to grayscale and blur it slightly
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.GaussianBlur( inputImage, kernelSize, borderConstant=0)
        blurred = cv2.GaussianBlur(gray, self.thresholdKernelSize, 0)
        # apply basic thresholding -- the first parameter is the image
        # we want to threshold, the second value is is our threshold
        # check; if a pixel value is greater than our threshold, we set it to be *black, otherwise it is *white*
        (T, threshInv) = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY)
        # cv2.imshow("Threshold Binary Inverse", threshInv)
        return threshInv


    def adjust_gamma(self, image, gamma):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)


    def process_frame(self, image, gamma=1):
        binary = self.get_treshold_binary(image)
        adjusted = self.adjust_gamma(image, gamma)
        masked = cv2.bitwise_and(adjusted, adjusted, mask=binary)
        return masked
    

    def set_deexposure_ratio(self, int8Image, int32Image):

        # Select ROI
        r = cv2.selectROI(int8Image)
        # Crop image
        # int8ImCrop8 = int8Image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        int32ImCrop = int32Image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        # b8, g8, r8 = int8ImCrop8[:, :, 0], int8ImCrop8[:, :, 1], int8ImCrop8[:, :, 2]
        # b32, g32, r32 = int32ImCrop[:, :, 0], int32ImCrop[:, :, 1], int32ImCrop[:, :, 2]

        self.deexposureRatio = 255/int32ImCrop.max()
        print('Deexposure Ratio:', self.deexposureRatio)



    def adjust_exposure(self, previewGamma=1):
        int8Image = None
        int32Image = None
        defined = False

        for filename in os.listdir(self.folder):

            if filename.endswith('.png'):
                image = cv2.imread(self.folder + filename, cv2.COLOR_BGR2GRAY)
                int8Masked = self.process_frame(image, previewGamma)
                int32Masked = self.process_frame(image)
                if not defined:
                    int8Image = np.asarray( int8Masked, dtype="int32" )
                    int32Image = np.asarray( int32Masked, dtype="int32" )
                    defined = True
                else:
                    int8Image += np.asarray( int8Masked, dtype="int32" )
                    int8Image = int8Image.clip(0,255)
                    int32Image += np.asarray( int32Masked, dtype="int32" )
                print(filename)
            else:
                continue

            cv2.imshow('int8', int8Image.astype(np.uint8))            
            cv2.waitKey(1)

        cv2.destroyWindow('int8')
        self.set_deexposure_ratio(int8Image.astype(np.uint8), int32Image)


    def render(self, gamma=1):
        finalImage = None
        defined = False

        for filename in os.listdir(self.folder):

            if filename.endswith('.png'):
                image = cv2.imread(self.folder + filename, cv2.COLOR_BGR2GRAY)
                masked = self.process_frame(image)
                if not defined:
                    finalImage = np.asarray( masked, dtype="int32" )
                    defined = True
                else:
                    finalImage += np.asarray( masked, dtype="int32" )
                print(filename)
            else:
                continue

            cv2.imshow('int8', self.adjust_gamma((finalImage*self.deexposureRatio).clip(0,255).astype(np.uint8), gamma))            
            cv2.waitKey(1)
        
        cv2.imwrite('results/testimage' + str(int(time.time())) + '.png', self.adjust_gamma((finalImage*self.deexposureRatio).clip(0,255).astype(np.uint8), gamma))


if __name__ == "__main__":
    ule = UnfoldingLongExposure('photos/photos_short_test_400_5.6_100_200_blending_edited')
    ule.adjust_exposure(0.4)
    import pdb; pdb.set_trace()
    ule.render(0.5)






# def film():
#     # BGR

#     VIDEO = 'short_test.mp4'
#     threshold = 2
#     gamma = 0.6


#     finalimage = None
#     defined = False

#     cap = cv2.VideoCapture(VIDEO)

#     # Check if camera opened successfully
#     if (cap.isOpened()== False): 
#         print("Error opening video stream or file")
    
#     count = 0

#     # Read until video is completed
#     while(cap.isOpened()):
#     # Capture frame-by-frame
#         ret, image = cap.read()
#         if ret == True:
#             image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
#             binary = get_treshold_binary(image, threshold)
#             adjusted = adjust_gamma(image, gamma)
#             masked = cv2.bitwise_and(adjusted, adjusted, mask=binary)
#             if not defined:
#                 finalimage = masked
#                 defined = True
#             else:
#                 finalimage += masked
#             count += 1
#             print(count)
#         else:
#             break

#         cv2.imshow('Image', finalimage)
#         # Press Q on keyboard to  exit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cv2.imwrite('results/testimage' + str(int(time.time())) + '.png', finalimage)
#     cap.release()
#     cv2.destroyAllWindows()