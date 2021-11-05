package lab;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import java.util.ArrayList;
import java.util.List;

public class LabOneCV {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String[] args) {

        LabOneCV labOneCV = new LabOneCV();

        labOneCV.detectFace();

        Mat currentImage = labOneCV.detectFaceIndented();

        Mat borderImage = labOneCV.borderSelection(currentImage);

        Mat deleteSmallBorders = labOneCV.deleteSmallBorders(borderImage);

        Mat resultContourRed = labOneCV.contourReduction(deleteSmallBorders);

        labOneCV.dilate(resultContourRed);

        Mat bilateral = labOneCV.bilateral(currentImage);

        Mat contrast = labOneCV.contrast(currentImage);

        Mat gaussNormal = labOneCV.gaussNormalize(borderImage);

        labOneCV.finalFilter(bilateral, contrast, gaussNormal);
    }
    public Mat loadingImage(String path){
        return new Imgcodecs().imread(path);
    }

    public boolean saveImage(Mat image, String name){
        Imgcodecs imgcodecs = new Imgcodecs();
        return imgcodecs.imwrite(name + ".jpg", image);
    }

    public void detectFace(){
        String path = "D:\\Programming\\Java\\LabCV\\image\\putInRussia.jpg";
        Mat originalImage = loadingImage(path);
        MatOfRect face = new MatOfRect();

        CascadeClassifier cascadeClassifier = new CascadeClassifier();
        cascadeClassifier.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml");
        cascadeClassifier.detectMultiScale(originalImage,
                face,
                1.05,
                5,
                Objdetect.CASCADE_SCALE_IMAGE,
                new Size(30, 30)
        );
        Rect[] faces = face.toArray();

        Imgproc.rectangle(originalImage, faces[0].tl(), faces[0].br(), new Scalar(205, 0, 0), 2);

        saveImage(originalImage, "detectFace");
    }

    public Mat detectFaceIndented() {
        String path = "D:\\Programming\\Java\\LabCV\\image\\putInRussia.jpg";
        Mat imageOriginal = loadingImage(path);
        MatOfRect face = new MatOfRect();

        CascadeClassifier cascadeClassifier = new CascadeClassifier();
        cascadeClassifier.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml");
        cascadeClassifier.detectMultiScale(imageOriginal,
                face,
                1.05,
                5,
                Objdetect.CASCADE_SCALE_IMAGE,
                new Size(30, 30)
        );

        Rect[] faces = face.toArray();
        Integer widthPart = imageOriginal.width()/10;
        Integer heightPart = imageOriginal.height()/10;

        //Как-то сложно считается
        Mat rectangle = new Mat(imageOriginal, new Rect(
                faces[0].x - widthPart,
                faces[0].y - heightPart,
                faces[0].width + 2*widthPart,
                faces[0].height + 2*heightPart));

        saveImage(rectangle, "detectFaceIndented");
        return rectangle;
    }

    public Mat borderSelection(Mat originalImage){
        Mat result = new Mat();
        Imgproc.Canny(originalImage,result, 50, 100);
        saveImage(result, "border");
        return result;
    }

    public Mat deleteSmallBorders(Mat image){
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(image, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

        List<MatOfPoint> smallContours = new ArrayList<>();
        for(int i = 0; i < contours.size(); i++){
            MatOfPoint cont = contours.get(i);
            if(cont.height() < 10 && cont.width() < 10){
                smallContours.add(cont);
            }
        }

        Imgproc.drawContours(image, smallContours, -1, new Scalar(0, 0, 0));

        saveImage(image, "deleteSmallBorders");
        return image;
    }

    public Mat contourReduction(Mat image){
        Mat result = new Mat();
        Imgproc.medianBlur(image, result,1);
        saveImage(result, "contourReduction");
        return result;
    }

    public void dilate(Mat image){
        Mat result = new Mat();
        Imgproc.dilate(image, result, Imgproc.getStructuringElement(Imgproc.MORPH_DILATE, new Size(3,3)));
        saveImage(result, "dilate");
    }

    public Mat gaussNormalize(Mat image){
        Mat result = new Mat(image.rows(), image.cols(), image.type());
        Mat normal = new Mat(image.rows(), image.cols(), 6);

        Imgproc.GaussianBlur(image, result, new Size(5, 5),0);
        for(int i = 0; i < image.height(); i++){
            for(int j = 0; j < image.width(); j++){
                normal.put(i,j, result.get(i,j)[0]/255.0);
            }
        }
        return normal;
    }

    public Mat bilateral(Mat image){
        Mat result = new Mat();
        Imgproc.bilateralFilter(image, result, 10,100,100);
        saveImage(result, "bilateral");
        return result;
    }

    public Mat contrast(Mat image){
        Mat gaussian = new Mat();
        Imgproc.GaussianBlur(image, gaussian, new Size(5, 5),0, 0 );

        Mat image1 = new Mat(image.rows(), image.cols(), image.type());
        Mat imageMul = image.mul(gaussian);
        for(int i = 0; i < image.height(); i++){
            for(int j = 0; j < image.width(); j++){
                image1.put(i,j,
                        image.get(i,j)[0] - imageMul.get(i,j)[0],
                              image.get(i,j)[1] - imageMul.get(i,j)[1],
                              image.get(i,j)[2] - imageMul.get(i,j)[2]);
            }
        }

        Mat result = new Mat(image.rows(), image.cols(), image.type());
        int k = 10;
        int T = 80;
        for(int i = 0; i < image.height(); i++){
            for(int j = 0; j < image.width(); j++){
                if(Math.abs(image1.get(i,j)[0]) <= T && Math.abs(image1.get(i,j)[1]) <= T && Math.abs(image1.get(i,j)[2]) <= T){
                    result.put(i,j,
                                image.get(i,j)[0],
                                image.get(i,j)[1],
                                image.get(i,j)[2]);
                }else {
                    result.put(i,j,
                            image.get(i,j)[0]+k*image1.get(i,j)[0],
                            image.get(i,j)[1]+k*image1.get(i,j)[1],
                            image.get(i,j)[2]+k*image1.get(i,j)[2]);
                }
            }
        }
        saveImage(result, "contrast");
        return result;
    }

    public void finalFilter(Mat imageF1, Mat imageF2, Mat imageM){
        Mat result = new Mat(imageF1.rows(), imageF1.cols(), 21);
        for(int i = 0; i < result.height(); i++){
            for(int j = 0; j < result.width(); j++){
                double compZero = imageM.get(i,j)[0]*imageF2.get(i,j)[0] + (1.0 - imageM.get(i,j)[0])*imageF1.get(i,j)[0];
                double compOne = imageM.get(i,j)[0]*imageF2.get(i,j)[1] + (1.0 - imageM.get(i,j)[0])*imageF1.get(i,j)[1];
                double compTwo = imageM.get(i,j)[0]*imageF2.get(i,j)[2] + (1.0 - imageM.get(i,j)[0])*imageF1.get(i,j)[2];
                result.put(i,j, compZero, compOne, compTwo);
            }
        }
        saveImage(result, "finalFilter");
    }
}
