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

import android.os.Bundle;
import android.widget.ArrayAdapter;
import android.widget.GridView;
import android.widget.ListView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.util.ArrayList;

public class NumbersActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Set the content of the activity to use the activity_main.xml layout file
        setContentView(R.layout.word_list);
        // Create an array of words
        ArrayList<Word> words = new ArrayList<Word>();
        //words.add ("one")

        //String[] words = new String[10];
        words.add(new Word("one","lutti"));
        words.add(new Word("tow","ottiko"));
        words.add(new Word("three","toloo"));
        words.add(new Word("four","oyssia"));
        words.add(new Word("five","massoka"));

        //Log.v("numbersActivity","word ad index 0: "+ words.get(0));
        WordAdapter adapter=new WordAdapter(this,words);
        // Find the View that shows the numbers category

        ListView listView = (ListView) findViewById(R.id.list_item);

        listView.setAdapter(adapter);
    }
}

