package com.example.montage;

import android.content.Context;
import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.IOException;
import java.util.List;

/** A basic Camera preview class */
public class CameraPreview extends SurfaceView implements SurfaceHolder.Callback {
    private static final String TAG = CameraPreview.class.getName();
    private Camera camera;
    private SurfaceHolder surfaceHolder;

    public CameraPreview(Context context, Camera camera) {
        super(context);
        this.camera = camera;
        surfaceHolder = getHolder();
        surfaceHolder.addCallback(this);
        surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
    }


    public void surfaceCreated(SurfaceHolder holder) {
        // Instruct the camera where to draw the preview
        try {
            camera.setPreviewDisplay(surfaceHolder);
            camera.startPreview();
        } catch (IOException e) {
            Log.d(TAG, "Error setting camera preview: " + e.getMessage());
        }
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        // Release the camera
        try {
            camera.release();
        } catch (Exception ignored) {
            Log.d(TAG, "Error releasing the camera");
        }
    }

    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
        // Check if the preview surface exists
        if (this.surfaceHolder.getSurface() == null) {
            return;
        }
        // Try to stop the preview before making any changes
        try {
            camera.stopPreview();
        } catch (Exception ignored) {

        }
        // Set the preview size
        Camera.Parameters parameters = camera.getParameters();
        Camera.Size bestSize = null;
        List<Camera.Size> sizeList = camera.getParameters().getSupportedPreviewSizes();
        bestSize = sizeList.get(0);
        for(int i = 1; i < sizeList.size(); i++){
            if((sizeList.get(i).width * sizeList.get(i).height) >
                    (bestSize.width * bestSize.height)){
                bestSize = sizeList.get(i);
            }
        }
        parameters.setRecordingHint(true);
        parameters.setPreviewSize(bestSize.width, bestSize.height);
        camera.setParameters(parameters);
        // Start the preview with the new menu
        try {
            camera.setPreviewDisplay(this.surfaceHolder);
            camera.startPreview();

        } catch (Exception e){
            Log.d(TAG, "Error starting camera preview: " + e.getMessage());
        }
    }
}