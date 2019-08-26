package com.example.montage;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

import androidx.fragment.app.Fragment;
import androidx.preference.PreferenceFragmentCompat;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * A simple {@link Fragment} subclass.
 */
public class SettingsFragment extends PreferenceFragmentCompat implements SharedPreferences.OnSharedPreferenceChangeListener {
    private static final String TAG = SettingsFragment.class.getName();

    @Override
    public void onCreatePreferences(Bundle savedInstanceState, String rootKey) {
        setPreferencesFromResource(R.xml.preferences, rootKey);
    }

    @Override
    public void onResume() {
        super.onResume();
        getPreferenceScreen().getSharedPreferences()
                .registerOnSharedPreferenceChangeListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        getPreferenceScreen().getSharedPreferences()
                .unregisterOnSharedPreferenceChangeListener(this);
    }

    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key)
    {
        String startDateString = sharedPreferences.getString(SettingsActivity.KEY_PREF_START_DATE, "").trim();
        String endDateString = sharedPreferences.getString(SettingsActivity.KEY_PREF_END_DATE, "").trim();
        System.out.println("startDate: " + startDateString);
        System.out.println("endDate: " + endDateString);
        if (!startDateString.equals("") && !endDateString.equals("")) {
            try {
                Date startDate = new SimpleDateFormat("MM-dd-yyyy").parse(startDateString);
                Date endDate = new SimpleDateFormat("MM-dd-yyyy").parse(endDateString);
                if (startDate.after(endDate)) {
                    Toast.makeText(getActivity(), "The start date is after the end date.", Toast.LENGTH_LONG).show();
                } else {
                    MainActivity.updateVideos(startDate, endDate);
                }
            } catch (ParseException e) {
                Toast.makeText(getActivity(), "Please follow the given formatting.", Toast.LENGTH_LONG).show();
            }
        }
        Log.d(TAG, "Unable to get string from preferences");
    }
}
