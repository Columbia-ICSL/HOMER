package com.example.montage;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.SharedPreferences;
import android.hardware.Camera;
import android.media.MediaPlayer;
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
import android.widget.VideoView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * MainActivity represents the core application activity.
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getName();
    public static final int MEDIA_TYPE_IMAGE = 1;
    public static final int MEDIA_TYPE_VIDEO = 2;
    public static final int FRONT_CAMERA_ID = 1;
    public static final int FILES_REQUEST_CODE = 42;
    private static final int HIGH_QUALITY_WIDTH = 1920;
    private static final int HIGH_QUALITY_HEIGHT = 1080;
    private static final int FRONT_FPS = 24;

    // private final String SERVER_IP = "209.2.214.30"; // IP of my laptop
    private final String SERVER_IP = "23.239.22.55"; // IP of ICSL server
    // Changed the port numbers to not conflict with other server file
    private static final int BACK_PORT = 8274;
    private static final int FRONT_PORT = 8933;
    private static final int CONTROL_PORT = 8186;
    private static final int HIGHLIGHT_PORT = 8844;
    private static final int BUF_LEN = 1024;
    // DCIM/Montage is a temporary folder for the montage source videos
    private static final String MEDIA_FOLDER = "Montage";

    private Camera camera;
    private CameraPreview cameraPreview;
    private MediaRecorder mediaRecorder;
    private boolean isRecording = false;
    private String timeStamp;
    private String backVideoFilename;
    private String frontVideoFilename;
    private static List<String> videoFilenames = new ArrayList<>();
    private static int videoFilenamesIndex = -1;
    private Thread highlightDownloadThread;

    private SharedPreferences sharedPreferences;
    private Button recordButton;
    private VideoView videoView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Additional UI elements
        recordButton = findViewById(R.id.button_record);
        videoView = findViewById(R.id.video_view);
        videoView.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
            @Override
            public void onCompletion(MediaPlayer mediaPlayer) {
                if (isRecording) {
                    stopRecording();
                }
            }
        });

        // Shared preferences
        // Do not override user's saved values
        androidx.preference.PreferenceManager.setDefaultValues(this, R.xml.preferences, false);
        sharedPreferences = androidx.preference.PreferenceManager.getDefaultSharedPreferences(this);

        String startDateString = sharedPreferences.getString(SettingsActivity.KEY_PREF_START_DATE, "").trim();
        String endDateString = sharedPreferences.getString(SettingsActivity.KEY_PREF_END_DATE, "").trim();
        if (!startDateString.equals("") && !endDateString.equals("")) {
            try {
                // Add 1 to endDate to make the boundary inclusive
                Date startDate = new SimpleDateFormat("MM-dd-yyyy").parse(startDateString);
                Date endDate = new SimpleDateFormat("MM-dd-yyyy").parse(endDateString);
                Calendar calendar = Calendar.getInstance();
                calendar.setTime(endDate);
                calendar.add(Calendar.DATE, 1);
                endDate = calendar.getTime();
                if (startDate.before(endDate)) {
                    updateVideos(startDate, endDate);
                    Toast.makeText(this, "Detected " + videoFilenames.size() + " videos", Toast.LENGTH_LONG).show();
                    for (int i = 0; i < videoFilenames.size(); i++) {
                        System.out.println(videoFilenames.get(i));
                    }
                } else {
                    Toast.makeText(this, "Using all videos", Toast.LENGTH_LONG).show();
                    updateVideos();
                    Toast.makeText(this, "Detected " + videoFilenames.size() + " videos", Toast.LENGTH_LONG).show();
                }
            } catch (ParseException e) {
                Toast.makeText(this, "Using all videos", Toast.LENGTH_LONG).show();
                updateVideos();
                Toast.makeText(this, "Detected " + videoFilenames.size() + " videos", Toast.LENGTH_LONG).show();
            }
        }
    }

    /**
     * Initialize the front camera preview and video view.
     */
    @Override
    protected void onResume() {
        super.onResume();
        // Initialize the front camera preview first to place it on top
        // Front camera
        camera = getCameraInstance(FRONT_CAMERA_ID);
        cameraPreview = new CameraPreview(this, camera);
        FrameLayout frontPreview = findViewById(R.id.camera_preview);
        frontPreview.addView(cameraPreview);

        // Initialize thread to download highlights
        if (highlightDownloadThread == null || !highlightDownloadThread.isAlive()) {
            highlightDownloadThread = new Thread(new Runnable() {
                @Override
                public void run() {
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
                    try {
                        ServerSocket serverSocket = new ServerSocket(HIGHLIGHT_PORT);
                        while (true) {
                            Socket clientSocket = serverSocket.accept();
                            Thread receiveHighlightThread = new Thread(new ReceiveHighlightTask(clientSocket, mediaStorageDir));
                            receiveHighlightThread.start();
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
     * Sets the text of the recording button to the given title.
     * @param title the new text to be displayed on the recording button
     */
    private void setRecordButtonText(String title) {
        recordButton.setText(title);
    }

    /**
     * Stops recording from both the back and front camera.
     */
    private void stopRecording() {
        try {
            mediaRecorder.stop();
            releaseMediaRecorder();
            camera.lock();

            setRecordButtonText("Record");
            isRecording = false;

            // Upload the back and front camera video and the control menu
            Log.v(TAG, "Trying to upload camera video");
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
                        sendVideoData(backVideoFilename, BACK_PORT);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            });
            Thread frontUploadThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    try  {
                        sendVideoData(frontVideoFilename, FRONT_PORT);
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

            // Advance to the next video in the list if there is one
            if (videoFilenamesIndex < videoFilenames.size() - 1) {
                videoFilenamesIndex += 1;
                // Bring the record button back (after uploads) to play next video
                toggleVisibility(recordButton);
            } else {
                // Notify user that there are no more videos
                MainActivity.this.runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(MainActivity.this, "No More Videos", Toast.LENGTH_LONG).show();
                    }
                });
            }
        } catch (Exception ignored) {
            // Avoid crashing on repeated button pressing
        }
    }

    /**
     * Starts recording from the front camera and playing a video.
     */
    private void startRecording() {
        if (videoFilenames.size() > 0 && videoFilenamesIndex >= 0 &&
                videoFilenamesIndex < videoFilenames.size()) {
            // Set the video from the list of local videos
            videoView.setVideoPath(videoFilenames.get(videoFilenamesIndex));
            backVideoFilename = videoFilenames.get(videoFilenamesIndex);
            if (prepareVideoRecorder()) {
                videoView.start();
                mediaRecorder.start();
                // Hide the record button while recording
                toggleVisibility(recordButton);
                isRecording = true;
            } else {
                releaseMediaRecorder();
            }
        }
    }

    /**
     * Starts recording if the app was not already. Does not provide a 'stop'
     * functionality since the recording will stop automatically when the video
     * ends.
     * @param view
     */
    public void onRecordClick(View view) throws InterruptedException {
        if (!isRecording) {
            startRecording();
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
     * Prepare and configure the media recorder for the front camera.
     * @return true if successful and false otherwise
     */
    private boolean prepareVideoRecorder(){
        mediaRecorder = new MediaRecorder();
        // Unlock the front camera and set the MediaRecorder camera
        camera.unlock();
        mediaRecorder.setCamera(camera);
        // Set the input sources
        mediaRecorder.setVideoSource(MediaRecorder.VideoSource.CAMERA);
        // Set the output menu manually
        mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        mediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.MPEG_4_SP);
        mediaRecorder.setVideoFrameRate(FRONT_FPS);
        mediaRecorder.setVideoSize(HIGH_QUALITY_WIDTH, HIGH_QUALITY_HEIGHT);
        // Set the output file
        frontVideoFilename = Objects.requireNonNull(getOutputFile(MEDIA_TYPE_VIDEO)).toString();
        mediaRecorder.setOutputFile(frontVideoFilename);
        // Set the preview output
        mediaRecorder.setPreviewDisplay(cameraPreview.getHolder().getSurface());
        // Prepare the configured MediaRecorder
        try {
            mediaRecorder.prepare();
        } catch (IllegalStateException e) {
            Log.d(TAG, "IllegalStateException preparing MediaRecorder: " + e.getMessage());
            releaseMediaRecorder();
            return false;
        } catch (IOException e) {
            Log.d(TAG, "IOException preparing MediaRecorder: " + e.getMessage());
            releaseMediaRecorder();
            return false;
        }
        return true;
    }

    /**
     * Releases the media recorder and camera.
     */
    @Override
    protected void onPause() {
        releaseMediaRecorder();
        releaseCamera();
        highlightDownloadThread.interrupt();
        super.onPause();
    }

    /**
     * Releases the front camera.
     */
    private void releaseCamera() {
        if (camera != null) {
            camera.setPreviewCallback(null);
            cameraPreview.getHolder().removeCallback(cameraPreview);
            camera.release();
            camera = null;
        }
    }

    /**
     * Releases the media recorder for the front camera.
     */
    private void releaseMediaRecorder() {
        if (mediaRecorder != null) {
            // Clear the recorder configuration
            mediaRecorder.reset();
            // Release the recorder object
            mediaRecorder.release();
            mediaRecorder = null;
            // Lock the camera for later use
            camera.lock();
        }
    }
    /**
     * Creates the output file for the camera.
     * @param type MEDIA_TYPE_IMAGE for images
     *             MEDIA_TYPE_VIDEO for videos
     * @return null if there is no external storage mounted
     */
    @SuppressLint("SimpleDateFormat")
    private File getOutputFile(int type){
        // Check that the external storage is mounted
        if (!Environment.getExternalStorageState().equalsIgnoreCase(Environment.MEDIA_MOUNTED)) {
            return null;
        }
        String childDir = "Front Camera";
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
        String prefix = "FRONT_";
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
     * Sends a String with the control menu information (formatted as a
     * Python dictionary).
     */
    private void sendControlData() {
        try {
            InetAddress serverAddress = InetAddress.getByName(SERVER_IP);
            Socket clientSocket = new Socket(serverAddress, CONTROL_PORT);
            // Build up the control menu dictionary String
            String done = "False";
            if (videoFilenamesIndex >= videoFilenames.size() - 1) {
                done = "True";
            }
            String dict = "{\'montage\' : " +
                    "True" +
                    ", \'done\' : " +
                    done +
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
     * ReceiveHighlightTask represents the highlight download.
     */
    private class ReceiveHighlightTask implements Runnable {
        private final Socket clientSocket;
        private final File mediaStorageDir;

        private ReceiveHighlightTask(Socket clientSocket, File mediaStorageDir) {
            this.clientSocket = clientSocket;
            this.mediaStorageDir = mediaStorageDir;
        }

        /**
         * Receives the highlight data from the server.
         */
        @Override
        public void run() {
            try {
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
                String in_filepath = mediaStorageDir.getAbsolutePath() + "/montage_" + timeStamp + ".mp4";
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
            } catch (IOException e) {
                System.out.println("IOException: " + e.getMessage());
            }
        }
    }


    /**
     * Toggles the visibility of a View.
     * @param view the view to toggle the visibility of
     */
    private void toggleVisibility(View view) {
        if (view.getVisibility() == View.GONE) {
            view.setVisibility(View.VISIBLE);
        } else if (view.getVisibility() == View.VISIBLE) {
            view.setVisibility(View.GONE);
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

    /**
     * Scans the local files and updates the video queue to those between
     * startDate and endDate inclusive.
     * @param startDate the start date
     * @param endDate the end date
     */
    public static void updateVideos(Date startDate, Date endDate) {
        // Scan local media
        if (!Environment.getExternalStorageState().equalsIgnoreCase(Environment.MEDIA_MOUNTED)) {
            return;
        }
        File localMediaDir = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DCIM), MEDIA_FOLDER);
        if (localMediaDir.isDirectory()) {
            videoFilenames = new ArrayList<>();
            for (File file : Objects.requireNonNull(localMediaDir.listFiles())) {
                if (file.isFile()) {
                    Date fileDate = new Date(file.lastModified());
                    if ((fileDate.equals(startDate) || fileDate.after(startDate)) &&
                            (fileDate.equals(endDate) || fileDate.before(endDate))) {
                        videoFilenames.add(file.getAbsolutePath());
                        System.out.println(file.getAbsoluteFile());
                        System.out.println(new SimpleDateFormat("yyyyMMdd_HHmmss").format(file.lastModified()));
                    }
                }
            }
        }
        if (videoFilenames.size() > 0) {
            videoFilenamesIndex = 0;
        }
    }

    /**
     * Scans the local files and updates the video queue with all videos.
     */
    public static void updateVideos() {
        // Scan local media
        if (!Environment.getExternalStorageState().equalsIgnoreCase(Environment.MEDIA_MOUNTED)) {
            return;
        }
        File localMediaDir = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DCIM), MEDIA_FOLDER);
        if (localMediaDir.isDirectory()) {
            for (File file : Objects.requireNonNull(localMediaDir.listFiles())) {
                if (file.isFile()) {
                    videoFilenames.add(file.getAbsolutePath());
                    System.out.println(file.getAbsoluteFile());
                    System.out.println(new SimpleDateFormat("yyyyMMdd_HHmmss").format(file.lastModified()));
                }
            }
        }
        if (videoFilenames.size() > 0) {
            videoFilenamesIndex = 0;
        }
    }
}
