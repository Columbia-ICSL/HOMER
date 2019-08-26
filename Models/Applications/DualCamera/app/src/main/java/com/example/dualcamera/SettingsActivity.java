package com.example.dualcamera;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;

public class SettingsActivity extends AppCompatActivity {
    public static final String KEY_PREF_SWITCH = "switch";
    public static final String KEY_PREF_MIN_DUR = "min_duration";
    public static final String KEY_PREF_MAX_DUR = "max_duration";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getSupportFragmentManager().beginTransaction()
                .replace(android.R.id.content, new SettingsFragment())
                .commit();
    }
}
