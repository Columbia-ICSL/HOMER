package com.example.dualcamera;

import android.content.Context;
import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.IOException;
import java.util.List;

public class FrontCameraPreview extends SurfaceView implements SurfaceHolder.Callback {
    private static final String TAG = FrontCameraPreview.class.getName();
    private Camera frontCamera;
    private SurfaceHolder surfaceHolder;

    public FrontCameraPreview(Context context, Camera camera) {
        super(context);
        frontCamera = camera;
        surfaceHolder = getHolder();
        surfaceHolder.addCallback(this);
        surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
    }

    @Override
    public void surfaceCreated(SurfaceHolder surfaceHolder) {
        // Instruct the back camera where to draw the preview
        try {
            frontCamera.setPreviewDisplay(surfaceHolder);
            frontCamera.startPreview();
        } catch (IOException e) {
            Log.d(TAG, "Error setting back camera preview: " + e.getMessage());
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder surfaceHolder, int format, int w, int h) {
        // Check if the preview surface exists
        if (this.surfaceHolder.getSurface() == null) {
            return;
        }
        // Try to stop the preview before making any changes
        try {
            frontCamera.stopPreview();
        } catch (Exception ignored) {

        }
        // Set the preview size
        Camera.Parameters parameters = frontCamera.getParameters();
        Camera.Size bestSize = null;
        List<Camera.Size> sizeList = frontCamera.getParameters().getSupportedPreviewSizes();
        bestSize = sizeList.get(0);
        for(int i = 1; i < sizeList.size(); i++){
            if((sizeList.get(i).width * sizeList.get(i).height) >
                    (bestSize.width * bestSize.height)){
                bestSize = sizeList.get(i);
            }
        }
        parameters.setRecordingHint(true);
        parameters.setPreviewSize(bestSize.width, bestSize.height);
        frontCamera.setParameters(parameters);
        // Start the preview with the new menu
        try {
            frontCamera.setPreviewDisplay(this.surfaceHolder);
            frontCamera.startPreview();

        } catch (Exception e){
            Log.d(TAG, "Error starting camera preview: " + e.getMessage());
        }
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
        // Release the camera
        try {
            frontCamera.release();
        } catch (Exception ignored) {
            Log.d(TAG, "Error releasing the front camera");
        }
    }
}
