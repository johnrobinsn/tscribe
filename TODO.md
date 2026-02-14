2.0 Feature Candidates
- [X] record/play  --start  and --end parameters to start and stop automatically
    * --start_time (can specify time for today or tomorrow given current time... or date/time)
    * --end_time
    * --start_delay (delays recording start for that duration can specify weeks, days, minutes, seconds)
    * --end_duration (ends recording after duration criteria met)
- [X] record --vad for voice audio detection (only record when voice is detected)
    * adds silence detection (configurable level)
    * uses silero vad to only record chunks with voice activity
- [X] alias command `list` to `ls`
- [X] transcribe overlapped with recording
 * reduce overall time to get to transcription by overlapping transcription activity.  ok to have transcription slightly delayed to deal make sure no submitted segments are cut in half and to provide better context for transcription.
- [X] add the ability to tag recordings and search via tags

Uncommitted
- [ ] transcribe --diarize
- [ ] better examples of using with claude code (blog or x post)
- [ ] store recordings as .mp3?
- [ ] webui with `serve`
- [ ] add `rm` command to manage old recordings

