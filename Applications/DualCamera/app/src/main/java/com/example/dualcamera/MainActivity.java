package com.example.dualcamera;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.hardware.Camera;
import android.media.AudioManager;
import android.media.CamcorderProfile;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * MainActivity represents the core application activity.
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getName();
    public static final int MEDIA_TYPE_IMAGE = 1;
    public static final int MEDIA_TYPE_VIDEO = 2;
    public static final int FILES_REQUEST_CODE = 42;
    private static final int HIGH_QUALITY_WIDTH = 1920;
    private static final int HIGH_QUALITY_HEIGHT = 1080;
    private static final int FRONT_FPS = 24;
    public static final int BACK_CAMERA_ID = 0;
    public static final int FRONT_CAMERA_ID = 1;
    private static final String SERVER_IP = "23.239.22.55"; // IP of the ICSL server
    private static final int BACK_PORT = 8000;
    private static final int FRONT_PORT = 8888;
    private static final int CONTROL_PORT = 8080;
    private static final int HIGHLIGHT_PORT = 8800;
    private static final int BUF_LEN = 1024;

    private Camera backCamera;
    private Camera frontCamera;
    private BackCameraPreview backCameraPreview;
    private FrontCameraPreview frontCameraPreview;
    private MediaRecorder backMediaRecorder;
    private MediaRecorder frontMediaRecorder;
    private boolean isRecording = false;
    private String timeStamp;
    private String backVideoFilename;
    private String frontVideoFilename;
    private Thread highlightDownloadThread;

    private SharedPreferences sharedPreferences;
    private Button recordButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Additional UI elements
        recordButton = findViewById(R.id.button_record);

        // Shared preferences
        // Do not override user's saved values
        androidx.preference.PreferenceManager.setDefaultValues(this, R.xml.preferences, false);
        sharedPreferences = androidx.preference.PreferenceManager.getDefaultSharedPreferences(this);

        // Mute the media recorder sounds
        setMuteAll(true);
    }

    /**
     * Mutes all audio from media recorders.
     * @param mute
     */
    private void setMuteAll(boolean mute) {
        AudioManager manager = (AudioManager) getSystemService(Context.AUDIO_SERVICE);

        int[] streams = new int[] { AudioManager.STREAM_ALARM,
                AudioManager.STREAM_DTMF, AudioManager.STREAM_MUSIC,
                AudioManager.STREAM_RING, AudioManager.STREAM_SYSTEM,
                AudioManager.STREAM_VOICE_CALL };

        for (int stream : streams) {
            assert(manager != null);
            manager.setStreamVolume(stream, 0, 0);
        }
    }

    /**
     * Gets a Camera object corresponding to the given cameraId.
     * @param cameraId 0 for back camera
     *                 1 for front camera
     * @return null if the camera is unavailable and a Camera object otherwise
     */
    public static Camera getCameraInstance(int cameraId){
        Camera c = null;
        // Try to get a Camera instance corresponding to the given cameraId
        try {
            c = Camera.open(cameraId);
        }
        catch (Exception e){
            // Camera is not available (in use or does not exist)
            Log.e(TAG,"Camera " + cameraId + " not available! " + e.toString() );
        }
        return c;
    }

    /**
     * Creates the output file for the camera.
     * @param type MEDIA_TYPE_IMAGE for images
     *             MEDIA_TYPE_VIDEO for videos
     * @param isBack true if the output file corresponds to the back camera
     *               false if the output file corresponds to the front camera
     * @return null if there is no external storage mounted
     */
    @SuppressLint("SimpleDateFormat")
    private File getOutputFile(int type, boolean isBack) {
        // Check that the external storage is mounted
        if (!Environment.getExternalStorageState().equalsIgnoreCase(Environment.MEDIA_MOUNTED)) {
            return null;
        }
        String childDir;
        if (isBack) {
            childDir = "Back Camera";
        } else {
            childDir = "Front Camera";
        }
        File mediaStorageDir = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_PICTURES), childDir);
        // Create the storage directory if it does not exist
        if (!mediaStorageDir.exists()){
            if (!mediaStorageDir.mkdirs()) {
                Log.d(TAG, "Failed to create directory (" + childDir + ")");
                return null;
            }
        }
        // Create the output file
        timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        File mediaFile;
        // Determine the appropriate filename prefix
        String prefix;
        if (isBack) {
            prefix = "BACK_";
        } else {
            prefix = "FRONT_";
        }
        if (type == MEDIA_TYPE_IMAGE){
            mediaFile = new File(mediaStorageDir.getPath() + File.separator +
                    prefix + "IMG_"+ timeStamp + ".jpg");
        } else if(type == MEDIA_TYPE_VIDEO) {
            mediaFile = new File(mediaStorageDir.getPath() + File.separator +
                    prefix + "VID_"+ timeStamp + ".mp4");
        } else {
            return null;
        }
        return mediaFile;
    }

    /**
     * Release the media recorder for the back camera.
     */
    private void releaseBackMediaRecorder(){
        if (backMediaRecorder != null) {
            // Clear the recorder configuration
            backMediaRecorder.reset();
            // Release the recorder object
            backMediaRecorder.release();
            backMediaRecorder = null;
            // Lock the camera for later use
            backCamera.lock();
        }
    }

    /**
     * Prepare and configure the media recorder for the back camera.
     * @return true if successful and false otherwise
     */
    private boolean prepareBackRecorder() {
        backMediaRecorder = new MediaRecorder();
        // Unlock the back camera and set the MediaRecorder camera
        backCamera.unlock();
        backMediaRecorder.setCamera(backCamera);
        // Set the input sources
        backMediaRecorder.setAudioSource(MediaRecorder.AudioSource.CAMCORDER);
        backMediaRecorder.setVideoSource(MediaRecorder.VideoSource.CAMERA);
        // Set the output menu with a Camcorder profile
        backMediaRecorder.setProfile(CamcorderProfile.get(CamcorderProfile.QUALITY_720P));
        // Set the output file
        backVideoFilename = Objects.requireNonNull(getOutputFile(MEDIA_TYPE_VIDEO, true)).toString();
        backMediaRecorder.setOutputFile(backVideoFilename);
        // Set the preview output
        backMediaRecorder.setPreviewDisplay(backCameraPreview.getHolder().getSurface());
        // Prepare the configured MediaRecorder
        try {
            backMediaRecorder.prepare();
        } catch (IllegalStateException e) {
            Log.d(TAG, "IllegalStateException preparing MediaRecorder: " + e.getMessage());
            releaseBackMediaRecorder();
            return false;
        } catch (IOException e) {
            Log.d(TAG, "IOException preparing MediaRecorder: " + e.getMessage());
            releaseBackMediaRecorder();
            return false;
        }
        return true;
    }

    /**
     * Release the media recorder for the front camera.
     */
    private void releaseFrontMediaRecorder() {
        if (frontMediaRecorder != null) {
            // Clear the recorder configuration
            frontMediaRecorder.reset();
            // Release the recorder object
            frontMediaRecorder.release();
            frontMediaRecorder = null;
            // Lock the camera for later use
            frontCamera.lock();
        }
    }

    /**
     * Prepare and configure the media recorder for the front camera.
     * @return true if successful and false otherwise
     */
    private boolean prepareFrontRecorder() {
        frontMediaRecorder = new MediaRecorder();
        // Unlock the front camera and set the MediaRecorder camera
        frontCamera.unlock();
        frontMediaRecorder.setCamera(frontCamera);
        // Set the input source (no audio)
        frontMediaRecorder.setVideoSource(MediaRecorder.VideoSource.CAMERA);
        // Set the output menu manually
        frontMediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        frontMediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.MPEG_4_SP);
        frontMediaRecorder.setVideoFrameRate(FRONT_FPS);
        frontMediaRecorder.setVideoSize(HIGH_QUALITY_WIDTH, HIGH_QUALITY_HEIGHT);
        // Set the output file
        frontVideoFilename = Objects.requireNonNull(getOutputFile(MEDIA_TYPE_VIDEO, false)).toString();
        frontMediaRecorder.setOutputFile(frontVideoFilename);
        // Set the preview output
        frontMediaRecorder.setPreviewDisplay(frontCameraPreview.getHolder().getSurface());
        // Prepare the configured MediaRecorder
        try {
            frontMediaRecorder.prepare();
        } catch (IllegalStateException e) {
            Log.d(TAG, "IllegalStateException preparing MediaRecorder: " + e.getMessage());
            releaseFrontMediaRecorder();
            return false;
        } catch (IOException e) {
            Log.d(TAG, "IOException preparing MediaRecorder: " + e.getMessage());
            releaseFrontMediaRecorder();
            return false;
        }
        return true;
    }

    /**
     * Releases the back camera.
     */
    private void releaseBackCamera() {
        if (backCamera != null) {
            backCamera.setPreviewCallback(null);
            backCameraPreview.getHolder().removeCallback(backCameraPreview);
            backCamera.release();
            backCamera = null;
        }
    }

    /**
     * Releases the front camera.
     */
    private void releaseFrontCamera() {
        if (frontCamera != null) {
            frontCamera.setPreviewCallback(null);
            frontCameraPreview.getHolder().removeCallback(frontCameraPreview);
            frontCamera.release();
            frontCamera = null;
        }
    }

    /**
     * Release the media recorders and cameras on pause.
     */
    @Override
    protected void onPause() {
        releaseBackMediaRecorder();
        releaseBackCamera();
        releaseFrontMediaRecorder();
        releaseFrontCamera();
        highlightDownloadThread.interrupt();
        super.onPause();
    }

    /**
     * Sets the text of the recording button to the given title.
     * @param title the new text to be displayed on the recording button
     */
    private void setRecordButtonText(String title) {
        recordButton.setText(title);
    }

    /**
     * Starts recording from both the back and front camera.
     */
    private void startRecording() {
        // Initialize the back and front media recorder
        if (prepareBackRecorder() && prepareFrontRecorder()) {
            backMediaRecorder.start();
            frontMediaRecorder.start();
            setRecordButtonText("Stop");
            isRecording = true;
        } else {
            releaseBackMediaRecorder();
            releaseFrontMediaRecorder();
        }
    }

    /**
     * Stops recording from both the back and front camera.
     */
    private void stopRecording() {
        try {
            backMediaRecorder.stop();
            frontMediaRecorder.stop();
            releaseBackMediaRecorder();
            releaseFrontMediaRecorder();
            backCamera.lock();
            frontCamera.lock();

            setRecordButtonText("Record");
            isRecording = false;

            // Upload the back and front camera video and the control menu
            Log.v(TAG, "Trying to upload camera videos");
            Thread controlUploadThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    try  {
                        sendControlData();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            });
            Thread backUploadThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    try  {
                        sendBackVideoData();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            });
            Thread frontUploadThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    try  {
                        sendFrontVideoData();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            });
            controlUploadThread.setPriority(Thread.MAX_PRIORITY);
            backUploadThread.setPriority(Thread.MAX_PRIORITY);
            frontUploadThread.setPriority(Thread.MAX_PRIORITY);
            controlUploadThread.start();
            backUploadThread.start();
            frontUploadThread.start();

            controlUploadThread.join();
            backUploadThread.join();
            frontUploadThread.join();
        } catch (Exception ignored) {
            // Avoid crashing on repeated button pressing
        }
    }

    /**
     * Called when the 'Record' button is clicked.
     * @param view
     */
    public void onRecordClick(View view) {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }

    /**
     * Initialize the front and back cameras with their respective previews.
     */
    @Override
    protected void onResume() {
        super.onResume();
        // Initialize the front camera preview first to place it on top
        // Front camera
        frontCamera = getCameraInstance(FRONT_CAMERA_ID);
        frontCameraPreview = new FrontCameraPreview(this, frontCamera);
        FrameLayout frontPreview = findViewById(R.id.front_camera_preview);
        frontPreview.addView(frontCameraPreview);

        // Back camera
        backCamera = getCameraInstance(BACK_CAMERA_ID);
        backCameraPreview = new BackCameraPreview(this, backCamera);
        FrameLayout backPreview = findViewById(R.id.back_camera_preview);
        backPreview.addView(backCameraPreview);

        // Initialize thread to download highlights
        if (highlightDownloadThread == null || !highlightDownloadThread.isAlive()) {
            highlightDownloadThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        while (true) {
                            receiveHighlightData();
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            });
            highlightDownloadThread.setPriority(Thread.NORM_PRIORITY);
            highlightDownloadThread.start();
        }
    }

    /**
     * Sends the video data corresponding to the given filename to the server
     * at the given port number.
     * @param filename the filename of the video to send
     * @param port the port number to connect to
     */
    private void sendVideoData(String filename, int port) {
        // Get the file to send
        File sendFile = new File(filename);
        if (!sendFile.exists()) {
            Log.d(TAG, "No video found");
            return;
        }
        String filepath = sendFile.getAbsolutePath();
        // Send the file
        try {
            InetAddress serverAddress = InetAddress.getByName(SERVER_IP);
            Socket clientSocket = new Socket(serverAddress, port);
            OutputStream outputStream = clientSocket.getOutputStream();
            System.out.println("Connection established");
            System.out.printf("Sending %s\n", filepath);
            FileInputStream fileInputStream = new FileInputStream(filepath);
            byte[] buffer = new byte[BUF_LEN];
            int readBytes;
            while ((readBytes = fileInputStream.read(buffer, 0, BUF_LEN)) != -1) {
                outputStream.write(buffer, 0, readBytes);
                System.out.printf("Sent %d bytes\n", readBytes);
            }
            System.out.println("Finished sending data");
            outputStream.flush();
            outputStream.close();
            clientSocket.close();
            System.out.println("Connection closed");
        } catch (IOException e) {
            Log.d(TAG, "IOException: " + e.getMessage());
        }
    }

    /**
     * Sends the back camera video to the server through the appropriate port.
     */
    private void sendBackVideoData() {
        sendVideoData(backVideoFilename, BACK_PORT);
    }

    /**
     * Sends the front camera video to the server through the appropriate port.
     */
    private void sendFrontVideoData() {
        sendVideoData(frontVideoFilename, FRONT_PORT);
    }

    /**
     * Sends a String with the control menu information (formatted as a
     * Python dictionary).
     */
    private void sendControlData() {
        try {
            InetAddress serverAddress = InetAddress.getByName(SERVER_IP);
            Socket clientSocket = new Socket(serverAddress, CONTROL_PORT);
            // Get user preference from settings menu
            // If automatic highlight duration detection is toggled
            AtomicInteger minDuration = new AtomicInteger();
            AtomicInteger maxDuration = new AtomicInteger();
            if (sharedPreferences.getBoolean(SettingsActivity.KEY_PREF_SWITCH, false)) {
                minDuration.set(-1);
                maxDuration.set(-1);
            }
            // Otherwise, read user input
            else {
                try {
                    minDuration.set(Integer.parseInt(sharedPreferences.getString(SettingsActivity.KEY_PREF_MIN_DUR, "-1")));
                } catch (NumberFormatException e) {
                    Log.d(TAG, "Invalid min highlight duration");
                    minDuration.set(-1);
                }
                try {
                    maxDuration.set(Integer.parseInt(sharedPreferences.getString(SettingsActivity.KEY_PREF_MAX_DUR, "-1")));
                } catch (NumberFormatException e) {
                    Log.d(TAG, "Invalid max highlight duration");
                    maxDuration.set(-1);
                }
            }
            // Build up the control menu dictionary String
            String dict = "{\'min_duration\' : " +
                    minDuration +
                    ", \'max_duration\' : " +
                    maxDuration +
                    ", \'montage\' : False" +
                    "}";
            System.out.println(dict);
            OutputStreamWriter outputStreamWriter = new OutputStreamWriter(clientSocket.getOutputStream(), StandardCharsets.UTF_8);
            outputStreamWriter.write(dict, 0, dict.length());
            outputStreamWriter.flush();
            outputStreamWriter.close();
            clientSocket.close();
        } catch (IOException e) {
            Log.d(TAG, "IOException: " + e.getMessage());
        }
    }

    /**
     * Receives the highlight data from the server.
     */
    private void receiveHighlightData() {
        // Check that the external storage is mounted
        if (!Environment.getExternalStorageState().equalsIgnoreCase(Environment.MEDIA_MOUNTED)) {
            Log.d(TAG, "Failed to mount external storage");
            return;
        }
        File mediaStorageDir = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_PICTURES), "Highlights");
        // Create the storage directory if it does not exist
        if (!mediaStorageDir.exists()) {
            if (!mediaStorageDir.mkdirs()) {
                Log.d(TAG, "Failed to create directory (Highlights)");
                return;
            }
        }
        // Try to connect to the server to download the highlight
        System.out.println("Waiting for connection");
        while (true) {
            try (
                    ServerSocket serverSocket = new ServerSocket(HIGHLIGHT_PORT);
                    Socket clientSocket = serverSocket.accept();
            ) {
                MainActivity.this.runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(MainActivity.this, "Highlight Downloaded", Toast.LENGTH_LONG).show();
                    }
                });
                System.out.println("Socket created");
                System.out.printf("Connected with %s: %s\n",
                        clientSocket.getInetAddress().toString(),
                        clientSocket.getLocalAddress().toString());
                @SuppressLint("SimpleDateFormat")
                String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
                String in_filepath = mediaStorageDir.getAbsolutePath() + "/highlight_" + timeStamp + ".mp4";
                File video = new File(in_filepath);
                FileOutputStream outputStream = new FileOutputStream(video);
                InputStream inputStream = clientSocket.getInputStream();
                byte[] buffer = new byte[BUF_LEN];
                int readBytes;
                while ((readBytes = inputStream.read(buffer)) != -1) {
                    System.out.printf("Received %d bytes\n", readBytes);
                    outputStream.write(buffer, 0, readBytes);
                }
                outputStream.close();
                inputStream.close();
                System.out.println("Finished receiving data");
                return;
            } catch (IOException e) {
                System.out.println("IOException: " + e.getMessage());
                return;
            }
        }
    }

    /**
     * Add 'Settings' into the option menu.
     * @param menu
     * @return
     */
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; adds items to the action bar if it is present
        getMenuInflater().inflate(R.menu.menu, menu);
        return true;
    }

    /**
     * Controls the actions followed by the selection of a menu item.
     * @param item the menu item selected
     * @return true if an item was selected and false otherwise
     */
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch(item.getItemId()) {
            case R.id.action_settings:
                Intent settingsIntent = new Intent(this, SettingsActivity.class);
                startActivity(settingsIntent);
                return true;
            case R.id.action_files:
                Intent filesIntent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                filesIntent.addCategory(Intent.CATEGORY_OPENABLE);
                filesIntent.setType("video/mp4");
                startActivityForResult(filesIntent, FILES_REQUEST_CODE);
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    /**
     * Controls the actions followed by the ending of an activity.
     * @param requestCode
     * @param resultCode
     * @param data
     */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == MainActivity.FILES_REQUEST_CODE) {
            if (data != null) {
                // Get the URI from the file intent
                Uri uri = data.getData();
                if (uri == null) {
                    return;
                }
                String uriPath = uri.getPath();
                if (uriPath == null) {
                    return;
                }
                File file;
                // Check if the file is from external storage
                if (uriPath.contains("/primary")) {
                    String[] uriPathSegments = uriPath.split(":");
                    String newPath = uriPathSegments[uriPathSegments.length - 1];
                    File sdCard = Environment.getExternalStorageDirectory();
                    file = new File(sdCard, newPath);
                    Log.i(TAG, "Showing file: " + file.getPath());
                    Intent videoIntent = new Intent(Intent.ACTION_VIEW, Uri.fromFile(file));
                    videoIntent.setDataAndType(Uri.fromFile(file), "video/*");
                    startActivity(videoIntent);
                }
                // Otherwise not supported
                else {
                    MainActivity.this.runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(MainActivity.this, "Not Permitted to View", Toast.LENGTH_LONG).show();
                        }
                    });
                }
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}
