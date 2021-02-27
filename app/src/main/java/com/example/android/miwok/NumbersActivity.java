/*
 * Copyright (C) 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.example.android.miwok;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.util.ArrayList;

public class NumbersActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Set the content of the activity to use the activity_main.xml layout file
        setContentView(R.layout.activity_numbers);
    // Create an array of words
        ArrayList<String> words=new ArrayList<String>();

        //String[] words = new String[10];
            words.add("one");
            words.add("two");
            words.add("three");
            words.add("four");
            words.add("five");
            words.add("six");
            words.add("seven");
            words.add("eight");
            words.add("nine");
            words.add("ten");
            Log.v("numbersActivity","word ad index 0: "+ words.get(0));
            Log.v("numbersActivity","word ad index 1: "+ words.get(1));
            Log.v("numbersActivity","word ad index 1: "+ words.get(2));
            Log.v("numbersActivity","word ad index 1: "+ words.get(3));
        // Find the View that shows the numbers category
        TextView numbers = (TextView) findViewById(R.id.numbers);

        LinearLayout rootView=(LinearLayout)findViewById(R.id.rootView);
        TextView wordView= new TextView(this);
        wordView.setText(words.get(0));
        rootView.addView(wordView);
    }
}

