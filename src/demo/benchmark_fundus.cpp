
#ifdef __OPENCL__
//this program requires opencl

#include "vglClImage.h"
#include "vglContext.h"
#include "cl2cpp_shaders.h"
#include "glsl2cpp_shaders.h"
#include "vglClFunctions.h"

#ifdef __OPENCV__
  #include <opencv2/imgproc/types_c.h>
  #include <opencv2/imgproc/imgproc_c.h>
  #include <opencv2/highgui/highgui_c.h>
#else
  #include <vglOpencv.h>
#endif
#include <chrono>
#include <iostream>
#include "demo/timer.h"

#include <fstream>
#include <string.h>


int main(int argc, char* argv[])
{

    float kernelData151[1][51] = {
        0.00037832, 0.00055477, 0.00080091, 0.00113832, 0.00159279, 0.00219416, 0.00297573, 0.00397312, 
        0.00522256, 0.0067585, 0.00861055, 0.01080005, 0.01333629, 0.0162128, 0.01940418, 0.02286371, 
        0.02652237, 0.0302895, 0.0340554, 0.03769589, 0.04107865, 0.04407096, 0.04654821, 0.04840248, 
        0.04955031, 0.04993894, 0.04955031, 0.04840248, 0.04654821, 0.04407096, 0.04107865, 0.03769589, 
        0.0340554, 0.0302895, 0.02652237, 0.02286371, 0.01940418, 0.0162128, 0.01333629, 0.01080005, 
        0.00861055, 0.0067585, 0.00522256, 0.00397312, 0.00297573, 0.00219416, 0.00159279, 0.00113832, 
        0.00080091, 0.00055477, 0.00037832
    };

    float kernelData511[51][1] = {
        0.00037832, 0.00055477, 0.00080091, 0.00113832, 0.00159279, 0.00219416, 0.00297573, 0.00397312, 
        0.00522256, 0.0067585, 0.00861055, 0.01080005, 0.01333629, 0.0162128, 0.01940418, 0.02286371, 
        0.02652237, 0.0302895, 0.0340554, 0.03769589, 0.04107865, 0.04407096, 0.04654821, 0.04840248, 
        0.04955031, 0.04993894, 0.04955031, 0.04840248, 0.04654821, 0.04407096, 0.04107865, 0.03769589, 
        0.0340554, 0.0302895, 0.02652237, 0.02286371, 0.01940418, 0.0162128, 0.01333629, 0.01080005, 
        0.00861055, 0.0067585, 0.00522256, 0.00397312, 0.00297573, 0.00219416, 0.00159279, 0.00113832, 
        0.00080091, 0.00055477, 0.00037832
    };

    float kernel151[1][51] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
    1.0f
    };
    float kernel511[1][51] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
    1.0f
    };


    float kernelData1717[17][17] = {
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}
    };



    // if (argc != 4)
    // {
    //     printf("\nUsage: demo_benchmark_cl lena_1024.tiff 1000 /tmp\n\n");
    //     printf("In this example, will run the program for lena_1024.tiff in a \nloop with 1000 iterations. Output images will be stored in /tmp.\n\n");
    //     printf("Error: Bad number of arguments = %d. 3 arguments required.\n", argc-1);
    //     exit(1);
    // }
    //vglInit(50,50);
    vglClInit();

    int nSteps = atoi(argv[2]);
    char* inFilename = argv[1];
    char* outPath = argv[3];
    char* outFilename = (char*) malloc(strlen(outPath) + 200);

    printf("VisionGL-OpenCL on %s, %d operations\n\n", inFilename, nSteps);
	
    printf("CREATING IMAGE\n");
    VglImage* img = vglLoadImage(inFilename, CV_LOAD_IMAGE_UNCHANGED, 0);

    printf("CHECKING NCHANNELS\n");
    if (img->nChannels == 3)
    {
        printf("NCHANNELS = 3\n");
        if (img->ndarray)
        {
            printf("NDARRAY not null\n");
            vglNdarray3To4Channels(img);
        }
        else
        {
            printf("NDARRAY IS null\n");
            vglIpl3To4Channels(img);
        }
    }
    img->vglShape->print();
    iplPrintImageInfo(img->ipl);

    printf("CHECKING IF IS NULL\n");
    if (img == NULL)
    {
        std::string str("Error: File not found: ");
        str.append(inFilename);
        printf("%s", str.c_str());
    }

 


    // Gray
    VglImage* gray = vglCreateImage(img);
    vglClRgb2Gray(img, gray);
    vglClFlush();
    printf("Gray done\n");
    int p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClRgb2Gray(img, gray);
    }
    vglClFlush();
    printf("Time spent on %8d Gray:                %s\n", nSteps, getTimeElapsedInMilliseconds());
 
    vglCheckContext(gray, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/out_cl_gray.tif");
    cvSaveImage(outFilename, gray->ipl);


    // Conv 1x51
    VglImage* conv_1x51 = vglCreateImage(gray);
    vglClConvolution(gray, conv_1x51 , (float*) kernelData151, 1, 51);
    vglClFlush();
    p = 0;
    TimerStart();
    while (p < nSteps) {
        p++;
        vglClConvolution(gray, conv_1x51, (float*)kernelData151, 1, 51);
    }
    vglClFlush();    
    printf("Time spent on %8d conv_1x51:                %s\n", nSteps, getTimeElapsedInMilliseconds());
    vglCheckContext(conv_1x51, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/out_cl_conv151.tif");
    cvSaveImage(outFilename, conv_1x51->ipl);


    // Conv 51x1
    VglImage* conv_51x1 = vglCreateImage(img);
    vglClConvolution(conv_1x51, conv_51x1 , (float*) kernelData151, 51, 1);
    vglClFlush();
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClConvolution(conv_1x51, conv_51x1 , (float*) kernelData151, 51, 1);
        
    }
    vglClFlush();
    printf("Time spent on %8d conv_51x1:                %s\n", nSteps, getTimeElapsedInMilliseconds());
    vglCheckContext(conv_51x1, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/out_cl_conv511.tif");
    cvSaveImage(outFilename, conv_51x1->ipl);


    // Dilate 1x51
    VglImage* dil_1x51 = vglCreateImage(img);
    vglClDilate(conv_51x1, dil_1x51, (float*) kernel151, 1, 51);
    vglClFlush();
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClDilate(conv_51x1, dil_1x51, (float*) kernel151, 1, 51);
    }
    vglClFlush();
    // printf("Dilate 1x51 done\n");
    printf("Time spent on %8d dil_1x51:                %s\n", nSteps, getTimeElapsedInMilliseconds());
    vglCheckContext(dil_1x51, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/out_cl_dil151.tif");
    cvSaveImage(outFilename, dil_1x51->ipl);


    // Dilate 51x1
    VglImage* dil_51x1 = vglCreateImage(img);
    vglClDilate(dil_1x51, dil_51x1, (float*) kernel511, 51, 1);
    vglClFlush();
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClDilate(dil_1x51, dil_51x1, (float*) kernel511, 51, 1);
    }
    vglClFlush();
    printf("Time spent on %8d dil_51x1:                %s\n", nSteps, getTimeElapsedInMilliseconds());
    // printf("Dilate 51x1 done\n");
    vglCheckContext(dil_51x1, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/out_cl_dil511.tif");
    cvSaveImage(outFilename, dil_51x1->ipl);


    // Erode 1x51
    VglImage* ero_1x51 = vglCreateImage(img);
    vglClErode(dil_51x1, ero_1x51, (float*) kernel151, 1, 51);
    vglClFlush();
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClErode(dil_51x1, ero_1x51, (float*) kernel151, 1, 51);
    }
    vglClFlush();
    printf("Time spent on %8d ero_1x51:                %s\n", nSteps, getTimeElapsedInMilliseconds());
    // printf("Erode 1x51 done\n");
    vglCheckContext(ero_1x51, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/out_cl_ero151.tif");
    cvSaveImage(outFilename, ero_1x51->ipl);


    // Erode 51x1
    VglImage* ero_51x1 = vglCreateImage(img);
    vglClErode(ero_1x51, ero_51x1, (float*) kernel511, 51, 1);
    vglClFlush();
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClErode(ero_1x51, ero_51x1, (float*) kernel511, 51, 1);
    }
    vglClFlush();
    printf("Time spent on %8d ero_51x1:                %s\n", nSteps, getTimeElapsedInMilliseconds());
    // printf("Erode 51x1 done\n");
    vglCheckContext(ero_51x1, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/out_cl_ero511.tif");
    cvSaveImage(outFilename, ero_51x1->ipl);


    // Sub
    VglImage* sub = vglCreateImage(img);
    vglClSub(ero_51x1, conv_51x1, sub);
    vglClFlush();
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClSub(ero_51x1, conv_51x1, sub);
    }
    vglClFlush();
    printf("Time spent on %8d sub:                %s\n", nSteps, getTimeElapsedInMilliseconds());
    // printf("Sub done\n");
    vglCheckContext(sub, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/out_cl_sub.tif");
    cvSaveImage(outFilename, sub->ipl);


    // Threshold
    VglImage* thresh = vglCreateImage(img);
    vglClThreshold(sub, thresh, 0.00784);
    vglClFlush();
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        vglClThreshold(sub, thresh, 0.00784);
    }
    vglClFlush();
    printf("Time spent on %8d thresh:                %s\n", nSteps, getTimeElapsedInMilliseconds());
    // printf("Threshold done\n");
    vglCheckContext(thresh, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/out_cl_thresh.tif");
    cvSaveImage(outFilename, thresh->ipl);



    // Rec
    VglImage* erod = vglCreateImage(img);
    VglImage* buffer = vglCreateImage(img);
    VglImage* buffer1 = vglCreateImage(img);
    VglImage* rec = vglCreateImage(img);

    vglClErode(thresh, erod, (float*) kernelData1717, 17, 17);
    printf("Erode done\n");
    // vglClConditionalDilate(erod, rec, buffer, (float*) kernelData1717, 17, 17);
    vglClReconstructionByDilation(erod, buffer, rec, buffer1, (float*) kernelData1717, 17, 17);
    vglClFlush();
    p = 0;
    TimerStart();
    while (p < nSteps)
    {
        p++;
        // vglClConditionalDilate(erod, rec, buffer, (float*) kernelData1717, 17, 17);
        vglClReconstructionByDilation(erod, buffer, rec, buffer1, (float*) kernelData1717, 17, 17);
    }
    vglClFlush();
    printf("Time spent on %8d rec:                %s\n", nSteps, getTimeElapsedInMilliseconds());
    // printf("Reconstruct done\n");
    vglCheckContext(rec, VGL_RAM_CONTEXT);
    sprintf(outFilename, "%s%s", outPath, "/out_cl_rec.tif");
    cvSaveImage(outFilename, rec->ipl);






    
    vglClFlush();
    return 0;

}

#endif
