import cv2
import numpy as np
import os
import time


class UnfoldingLongExposure:


    def __init__(self, inputFolder, outputFile=None, thresholdKernelSize=(7,7), threshold=2, imageType='tif', deexposureRatio=None):
        if inputFolder[-1] != '/':
            inputFolder += '/'
        self.folder = inputFolder
        self.output = outputFile
        self.thresholdKernelSize = thresholdKernelSize
        self.threshold = threshold
        self.imageType = '.' + imageType
        self.deexposureRatio = deexposureRatio


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
        if gamma == 1:
            return image
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
    

    def set_deexposure_ratio(self, int8Image, int32Image, toMaximum=False):

        if not toMaximum:
            # Select ROI
            r = cv2.selectROI(int8Image)
            int32ImCrop = int32Image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            self.deexposureRatio = 255/int32ImCrop.max()
            # b8, g8, r8 = int8ImCrop8[:, :, 0], int8ImCrop8[:, :, 1], int8ImCrop8[:, :, 2]
            # b32, g32, r32 = int32ImCrop[:, :, 0], int32ImCrop[:, :, 1], int32ImCrop[:, :, 2]
        else:
            self.deexposureRatio = 255/int32Image.max()

        print('Deexposure Ratio:', self.deexposureRatio)



    def adjust_exposure(self, previewGamma=1, visualize=False, toMaximum=False):
        int8Image = None
        int32Image = None
        defined = False

        for filename in os.listdir(self.folder):

            if filename.endswith(self.imageType):
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
            
            if visualize == True:
                cv2.imshow('int8', int8Image.astype(np.uint8))            
                cv2.waitKey(1)

        cv2.destroyWindow('int8')
        self.set_deexposure_ratio(int8Image.astype(np.uint8), int32Image, toMaximum)


    def renderImage(self, gamma=1):
        finalImage = None
        defined = False

        for filename in os.listdir(self.folder):

            if filename.endswith(self.imageType):
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
        
        cv2.imwrite('results/testimage' + str(int(time.time())) + self.imageType, self.adjust_gamma((finalImage*self.deexposureRatio).clip(0,255).astype(np.uint8), gamma))


    def renderVideo(self, gamma=1, showMoment=True, fadeOutFrames=3, final=False):
        finalImage = None
        fadeOutBuffer = []
        finalFrames = []

        fadeSum = sum((i+1) / (fadeOutFrames+1) for i in range(fadeOutFrames))
        momentIntensity = 1 / fadeSum

        for filename in os.listdir(self.folder):
            if filename.endswith(self.imageType):
                image = cv2.imread(self.folder + filename, cv2.COLOR_BGR2GRAY)
                if finalImage is None:
                    finalImage = np.zeros(image.shape, 'int32')
                masked = np.asarray(self.process_frame(image), dtype="int32")
                fadeOutBuffer.append(masked.copy())
                if len(fadeOutBuffer) > fadeOutFrames:
                    finalImage += fadeOutBuffer.pop(0)
                frame = finalImage.copy()
                exposed = (frame*self.deexposureRatio)
                if showMoment:
                    for i, fader in enumerate(fadeOutBuffer):
                        exposed += ((i+1)/(fadeOutFrames+1)) * momentIntensity * fader
                gammad = self.adjust_gamma(exposed, gamma)
                finalFrames.append(gammad.clip(0,255).astype(np.uint8))
                print(filename)
            else:
                continue

        if showMoment:  
            while fadeOutBuffer:
                frame = finalImage.copy()
                exposed = (frame*self.deexposureRatio)
                for i, fader in enumerate(fadeOutBuffer):
                    exposed += ((i+1)/(len(fadeOutBuffer)+1))  * momentIntensity * fader
                gammad = self.adjust_gamma(exposed, gamma)
                finalFrames.append(gammad.clip(0,255).astype(np.uint8))
                finalImage += fadeOutBuffer.pop(0)
        
        name = str(int(time.time())) + '_' + str(self.output) + '_' + str(self.deexposureRatio)

        if final:
            directory = 'results/' + name
            os.mkdir(directory)
            for i, frame in enumerate(finalFrames):
                    cv2.imwrite(directory + '/' + str(1000+i) + '.tif', frame)
        else:
            height, width, _ = finalImage.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter('results/' + name + '.mp4', fourcc, 25, (width, height))

            for frame in finalFrames:
                videoWriter.write(frame)
                cv2.imshow('int8', frame)
                cv2.waitKey(1)

            videoWriter.release()



if __name__ == "__main__":
    # ule = UnfoldingLongExposure('photos/photos_short_test_400_5.6_100_200_blending_edited', deexposureRatio=0.0092804891363686)
    # ule = UnfoldingLongExposure('photos/photos_shoot_1', deexposureRatio=0.07227891156462585) #0.07227891156462585
    # ule = UnfoldingLongExposure('photos/photos_shoot_4', deexposureRatio=0.0037648378905096556) #0.0037648378905096556
    
    
    ule = UnfoldingLongExposure('photos/photos_andrea_pf_test', imageType='tif')
    ule.adjust_exposure(toMaximum=True)
    ule.renderVideo(showMoment=True, fadeOutFrames=1)

    ule = UnfoldingLongExposure('photos/photos_luca_fuzz_test', imageType='tif')
    ule.adjust_exposure(toMaximum=True)
    ule.renderVideo(showMoment=True, fadeOutFrames=1)

    ule = UnfoldingLongExposure('photos/photos_andrea_div_test', imageType='tif', deexposureRatio=0.007447429906542056)
    ule.renderVideo(showMoment=True, fadeOutFrames=1)

    # ule = UnfoldingLongExposure('photos/photos_short_test', imageType='tif')
    # ule.adjust_exposure(toMaximum=True)
    # ule.renderVideo(showMoment=True, fadeOutFrames=1)

    # ule = UnfoldingLongExposure('photos/photos_short_test', imageType='tif')
    # ule.adjust_exposure(previewGamma=0.25, visualize=True, toMaximum=True)
    # ule.renderVideo(showMoment=True, fadeOutFrames=1)

    # ule = UnfoldingLongExposure('photos/photos_andrea_div_test', imageType='tif', deexposureRatio=0.007447429906542056)
    # ule.renderVideo(showMoment=True, fadeOutFrames=5)



    # ule.adjust_exposure(previewGamma=0.25, visualize=True, toMaximum=True)
    # # import pdb; pdb.set_trace()
    # # ule.renderImage()
    # ule.renderVideo(showMoment=True, fadeOutFrames=5)






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