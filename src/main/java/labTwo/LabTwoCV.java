package labTwo;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.concurrent.ThreadLocalRandom;

public class LabTwoCV {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    ArrayList<ArrayList<ArrayList<Integer>>> getKernelInteger(int sizeKernel, int startRange, int finishRange){

        ArrayList<ArrayList<ArrayList<Integer>>> kern = new ArrayList<>(sizeKernel);
        for(int i = 0; i < sizeKernel; i++){
            kern.add(new ArrayList<>(sizeKernel));
            for(int j = 0; j < sizeKernel; j++)
                kern.get(i).add(new ArrayList<>(sizeKernel));
        }

        for(int i = 0; i < sizeKernel; i++)
            for(int j = 0; j < sizeKernel; j++)
                for(int k = 0; k < sizeKernel; k++)
                    kern.get(i).get(j).add(ThreadLocalRandom.current().nextInt(startRange, finishRange));
        return kern;
    }

    Mat convolutions(Mat originalImage, ArrayList<ArrayList<ArrayList<Integer>>> kernel, int numberKernel){
        Mat conv = new Mat(originalImage.rows(), originalImage.cols(), 6);

        ArrayList<Mat> convPart = new ArrayList<>();
        for(int i = 0; i < 3; i++){
            convPart.add(new Mat(originalImage.rows(), originalImage.cols(), 6));
        }

        ArrayList<Mat> bgr = new ArrayList<>();
        Core.split(originalImage, bgr);

        for(int k = 0; k < 3; k++){
            Mat kernelMat = new Mat(3,3,1);
            ArrayList<ArrayList<Integer>> kernelCurrent = kernel.get(k);

            for(int i = 0; i < 3; i++){
                for(int j = 0; j < 3; j++){
                    kernelMat.put(i,j,kernelCurrent.get(i).get(j));
                }
            }
            Imgproc.filter2D(bgr.get(k),convPart.get(k),-1,kernelMat);
        }

        for(int i = 0; i < originalImage.rows(); i++)
            for(int j = 0; j < originalImage.cols(); j++)
                conv.put(i,j, convPart.get(0).get(i,j)[0] + convPart.get(1).get(i,j)[0] + convPart.get(2).get(i,j)[0]);

        saveImage(conv, "conv" + numberKernel);
        return conv;
    }

    Mat ReLU(int rows, int cols, Mat imageConvolutionsNormalized){
        Mat mask = new Mat(rows, cols, 6);
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                if(imageConvolutionsNormalized.get(i,j)[0] > 0.0)
                    mask.put(i,j,1);
                else
                    mask.put(i,j,0);

        return mask.mul(imageConvolutionsNormalized);
    }

    Mat MaxPooling(int rows, int cols, Mat imagesReLU){
        Mat maxPool = new Mat(rows - 1, cols - 1, 6);
        for(int i = 0; i < rows - 1; i++)
            for(int j = 0; j < cols - 1; j++){
                ArrayList<Double> pooling = new ArrayList<>();
                pooling.add(imagesReLU.get(i,j)[0]);
                pooling.add(imagesReLU.get(i+1,j)[0]);
                pooling.add(imagesReLU.get(i,j+1)[0]);
                pooling.add(imagesReLU.get(i+1,j+1)[0]);

                maxPool.put(i, j, Collections.max(pooling));
            }
        return maxPool;
    }

    ArrayList<Mat> softMax(int rows, int cols, ArrayList<Mat> imagesMaxPool){
        Mat mergeMat = new Mat();
        Core.merge(imagesMaxPool, mergeMat);
        Mat softMaxMat = new Mat(mergeMat.rows(), mergeMat.cols(), mergeMat.type());

//        System.out.println(softMax.rows());
//        System.out.println(softMax.cols());
//        System.out.println(softMax.type());

        for(int i = 0; i < rows - 1; i++) {
            for (int j = 0; j < cols - 1; j++) {
                Double sumExp = 0.0;
                for (int k = 0; k < 5; k++) {
                    sumExp += mergeMat.get(i, j)[k];
                }

                softMaxMat.put(i, j, mergeMat.get(i, j)[0] / sumExp,
                        mergeMat.get(i, j)[1] / sumExp, mergeMat.get(i, j)[2] / sumExp,
                        mergeMat.get(i, j)[3] / sumExp, mergeMat.get(i, j)[4] / sumExp);
            }
        }

        ArrayList<Mat> split = new ArrayList<>(5);
        Core.split(softMaxMat,split);

        return split;
    }

    public Mat loadingImage(String path){
        return new Imgcodecs().imread(path);
    }

    public boolean saveImage(Mat image, String name){
        Imgcodecs imgcodecs = new Imgcodecs();
        return imgcodecs.imwrite(name + ".jpg", image);
    }

    public static void main(String[] args) {
        LabTwoCV labTwoCV = new LabTwoCV();

        String path = "D:\\Programming\\Java\\LabCV\\image\\putInRussia.jpg";
        Mat originalImage = labTwoCV.loadingImage(path);

        //  ---------------     Convolutions     ---------------

        int sizeKernel = 3;
        int start = -20;
        int finish = 20;
        ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>> kernels = new ArrayList<>(5);
        for(int i = 0; i < 5; i ++){
            kernels.add(labTwoCV.getKernelInteger(sizeKernel, start, finish));
        }

        //  Print kernels
//        for(int p = 0; p < 5; p++){
//            for(int i = 0; i < sizeKernel; i++){
//                for(int j = 0; j < sizeKernel; j++){
//                    for(int k = 0; k < sizeKernel; k++)
//                        System.out.print(kernels.get(p).get(i).get(j).get(k) + " ");
//                    System.out.println();
//                }
//                System.out.println();
//            }
//            System.out.println("-------------------------");
//        }
        ArrayList<Mat> imageConvolutions = new ArrayList<>();
        for(int i = 0; i < 5; i++){
            imageConvolutions.add(labTwoCV.convolutions(originalImage, kernels.get(i), i+1));
        }

        //  ---------------     Normalize     ---------------

        ArrayList<Mat> imageConvolutionsNormalized = new ArrayList<>();
        for(int i = 0; i < 5; i++){
            imageConvolutionsNormalized.add(new Mat());
        }
        for(int i = 0; i < 5; i++){
            Core.normalize(imageConvolutions.get(i), imageConvolutionsNormalized.get(i), 0, 1, Core.NORM_MINMAX, 6);
        }

//        for(int i = 0; i < imageConvolutionsNormalized.get(0).rows(); i++){
//            for(int j = 0; j < imageConvolutionsNormalized.get(0).cols(); j++)
//                System.out.print(imageConvolutionsNormalized.get(1).get(i,j)[0] + " ");
//            System.out.println();
//        }


        //  ---------------     ReLu     ---------------
        ArrayList<Mat> imagesReLU = new ArrayList<>(5);
        for(int i = 0; i < 5; i++){
            imagesReLU.add(labTwoCV.ReLU(originalImage.rows(), originalImage.cols(),
                    imageConvolutionsNormalized.get(i)));
        }

        //  Print ReLU
/*        for(int i = 0; i < imageReLu.rows(); i++){
            for(int j = 0; j < imageReLu.cols(); j++)
                System.out.print(imageReLu.get(i,j)[0] + " ");
            System.out.println();
        }*/

        //  ---------------     MaxPooling     ---------------

        ArrayList<Mat> imagesMaxPool = new ArrayList<>(5);
        for(int i = 0; i < 5; i++){
            imagesMaxPool.add(labTwoCV.MaxPooling(originalImage.rows(),
                    originalImage.cols(),imagesReLU.get(i) ));
        }

//        for(int i = 0; i < originalImage.rows() - 1; i++){
//            for(int j = 0; j < originalImage.cols() - 1; j++){
//                System.out.print(imagesMaxPool.get(0).get(i, j)[0] + " ");
//            }
//            System.out.println();
//        }

        //  ---------------     SoftMax     ---------------

        ArrayList<Mat> imagesSoftMax = labTwoCV.softMax(originalImage.rows(),
                originalImage.cols(), imagesMaxPool);

//          Print part of softMax
//        for(int i = 0; i < originalImage.rows() - 1; i++){
//            for(int j = 0; j < originalImage.cols() - 1; j++){
//                System.out.print(imagesSoftMax.get(0).get(i, j)[0] + " ");
//            }
//            System.out.println();
//        }
    }
}
