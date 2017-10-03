#include<stdio.h>
#include<stddef.h>
#include<stdlib.h>
#include<unistd.h>
#include<ueye.h>

// Based on https://gist.github.com/ChristianUlbrich/7871863

// compile:
//$$$ gcc -Wall test_frame_capture.c -lueye_api -o frametest

HIDS hCam = 1;

int main() {
  //printf("Success-Code: %d\n",IS_SUCCESS);
  //Kamera öffnen  _>  Open the camera
  INT nRet = is_InitCamera (&hCam, NULL);
  //printf("Status Init %d\n",nRet);


  // // Get the exposure time
  // double nExposure;
  // nRet = is_Exposure (hCam, IS_EXPOSURE_CMD_GET_EXPOSURE,(void*)&nExposure, 8);
  // if (nRet == IS_SUCCESS){
  //   printf("Exposure time (ms) %f\n",nExposure);
  // }
  // else{
  //   printf("Error: Status is_Exposure %d\n",nRet);
  // }

  // Aquí eliges el valor del tiempo de exposición en segundos (si no puede elegir ese valor, elige uno cercano)
  double nExposureSet = 2;
  nRet = is_Exposure (hCam, IS_EXPOSURE_CMD_SET_EXPOSURE,(void*)&nExposureSet, 8);
  if (nRet == IS_SUCCESS){
      printf("Expo time (ms) %f. ",nExposureSet);
    }


  //
  // UINT nExposureSet = 100;
  // nRet = is_Exposure (hCam, IS_EXPOSURE_CMD_SET_EXPOSURE,
  //                       (void*)&nExposureSet,
  //                       sizeof(nExposureSet));
  // printf("Status is_Exposure %d\n",nRet);
  //
  // nRet = is_Exposure (hCam, IS_EXPOSURE_CMD_GET_EXPOSURE,
  //                       (void*)&nExposure,
  //                       sizeof(nExposure));
  // printf("Status is_Exposure %d\n",nRet);
  // printf("Status is_Exposure value (ms) %d\n",nExposure);




  // //Pixel-Clock setzen  _>  Pixel clock
  // UINT nPixelClockDefault = 9;
  // nRet = is_PixelClock(hCam, IS_PIXELCLOCK_CMD_SET,
  //                       (void*)&nPixelClockDefault,
  //                       sizeof(nPixelClockDefault));
  //
  // //printf("Status is_PixelClock %d\n",nRet);

  //Farbmodus der Kamera setzen  _>  Color mode of the camera
  //INT colorMode = IS_CM_CBYCRY_PACKED;
  //INT colorMode = IS_CM_BGR8_PACKED;
  INT colorMode = IS_CM_MONO8;


  nRet = is_SetColorMode(hCam,colorMode);
  //printf("Status SetColorMode %d\n",nRet);

  UINT formatID = 4;
  //Bildgröße einstellen  _>  Set the image size -> 1280x1024
  nRet = is_ImageFormat(hCam, IMGFRMT_CMD_SET_FORMAT, &formatID, 4);
  //printf("Status ImageFormat %d\n",nRet);

  //Speicher für Bild alloziieren  _>  Memory for picture
  char* pMem = NULL;
  int memID = 0;
  nRet = is_AllocImageMem(hCam, 1280, 1024, 8, &pMem, &memID);
  //printf("Status AllocImage %d\n",nRet);

  //diesen Speicher aktiv setzen  _>  Activate this memory
  nRet = is_SetImageMem(hCam, pMem, memID);
  //printf("Status SetImageMem %d\n",nRet);

  //Bilder im Kameraspeicher belassen  _>  Images in the camera memory left
  INT displayMode = IS_SET_DM_DIB;
  nRet = is_SetDisplayMode (hCam, displayMode);
  //printf("Status displayMode %d\n",nRet);

  //Bild aufnehmen  _>  Take picture
  nRet = is_FreezeVideo(hCam, IS_WAIT);
  //printf("Status is_FreezeVideo %d\n",nRet);


  //Directorio actual:
  char buf[256];
  wchar_t dir[256];
  size_t len;
  if ((len = readlink("/proc/self/exe", buf, sizeof(buf)-1)) != -1)
    //printf("Dir: %s \n",buf);
    buf[len-17] = '\0';
    //printf("Dir: %s \n",buf);
    swprintf(dir, sizeof(dir), L"%s%s", buf, "/snap_BGR8.png");

  //Bild aus dem Speicher auslesen und als Datei speichern  _>  Read the image from memory and save it as a file
  IMAGE_FILE_PARAMS ImageFileParams;
  //ImageFileParams.pwchFileName = L"/home/santiago/Documentos/MisPys/modulador/snap_BGR8.png";
  ImageFileParams.pwchFileName = dir;
  ImageFileParams.pnImageID = NULL;
  ImageFileParams.ppcImageMem = NULL;
  ImageFileParams.nQuality = 0;
  ImageFileParams.nFileType = IS_IMG_PNG;

  nRet = is_ImageFile(hCam, IS_IMAGE_FILE_CMD_SAVE, (void*) &ImageFileParams, sizeof(ImageFileParams));
  printf("Status %d. ",nRet);

  // ImageFileParams.pwchFileName = L"./snap_BGR8.bmp";
  // ImageFileParams.pnImageID = NULL;
  // ImageFileParams.ppcImageMem = NULL;
  // ImageFileParams.nQuality = 0;
  // ImageFileParams.nFileType = IS_IMG_BMP;
  //
  // nRet = is_ImageFile(hCam, IS_IMAGE_FILE_CMD_SAVE, (void*) &ImageFileParams, sizeof(ImageFileParams));
  // printf("Status is_ImageFile %d\n",nRet);
  //
  // ImageFileParams.pwchFileName = L"./snap_BGR8.jpg";
  // ImageFileParams.pnImageID = NULL;
  // ImageFileParams.ppcImageMem = NULL;
  // ImageFileParams.nQuality = 0;
  // ImageFileParams.nFileType = IS_IMG_JPG;
  //
  // nRet = is_ImageFile(hCam, IS_IMAGE_FILE_CMD_SAVE, (void*) &ImageFileParams, sizeof(ImageFileParams));
  // printf("Status is_ImageFile %d\n",nRet);

  //Kamera wieder freigeben  _>  Release the camera again
  is_ExitCamera(hCam);
}
