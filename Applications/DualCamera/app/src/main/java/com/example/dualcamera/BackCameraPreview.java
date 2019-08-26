package com.example.dualcamera;

import android.content.Context;
import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.IOException;
import java.util.List;

public class BackCameraPreview extends SurfaceView implements SurfaceHolder.Callback {
    private static final String TAG = BackCameraPreview.class.getName();
    private Camera backCamera;
    private SurfaceHolder surfaceHolder;

    public BackCameraPreview(Context context, Camera camera) {
        super(context);
        backCamera = camera;
        surfaceHolder = getHolder();
        surfaceHolder.addCallback(this);
        surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
    }

    @Override
    public void surfaceCreated(SurfaceHolder surfaceHolder) {
        // Instruct the back camera where to draw the preview
        try {
            backCamera.setPreviewDisplay(surfaceHolder);
            backCamera.startPreview();
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
            backCamera.stopPreview();
        } catch (Exception ignored) {

        }
        // Set the preview size
        Camera.Parameters parameters = backCamera.getParameters();
        Camera.Size bestSize = null;
        List<Camera.Size> sizeList = backCamera.getParameters().getSupportedPreviewSizes();
        bestSize = sizeList.get(0);
        for(int i = 1; i < sizeList.size(); i++){
            if((sizeList.get(i).width * sizeList.get(i).height) >
                    (bestSize.width * bestSize.height)){
                bestSize = sizeList.get(i);
            }
        }
        parameters.setRecordingHint(true);
        parameters.setPreviewSize(bestSize.width, bestSize.height);
        backCamera.setParameters(parameters);
        // Start the preview with the new menu
        try {
            backCamera.setPreviewDisplay(this.surfaceHolder);
            backCamera.startPreview();

        } catch (Exception e){
            Log.d(TAG, "Error starting camera preview: " + e.getMessage());
        }
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
        // Release the camera
        try {
            backCamera.release();
        } catch (Exception ignored) {
            Log.d(TAG, "Error releasing the back camera");
        }
    }
}
