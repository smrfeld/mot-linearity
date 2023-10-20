file_to_tracks = load_tracks(DataSpec())

# Plot lines for every track
# for tracks in file_to_tracks.values():
#     for track_id,track in tracks.tracks.items():
tracks = list(file_to_tracks.values())[3]
track_id,track = list(tracks.tracks.items())[4]

fig = plot_track(track)
# fig.show()

segments = find_linear_segments(track, tol=0.2)
print(f"Track {track_id} has {len(segments)} linear segments")
print(segments)

def add_lin_segments(fig: go.Figure, segments: List[LinSeg], track: Track):
    for seg in segments:
        for i,j in [(0,1),(2,3)]:
            x = [track.boxes[x].xyxy[i] for x in range(seg.idx_start_incl, seg.idx_end_incl+1)]
            y = [track.boxes[x].xyxy[j] for x in range(seg.idx_start_incl, seg.idx_end_incl+1)]
            plot_line(fig, x, y, col="red")

            x = [track.boxes[seg.idx_start_incl].xyxy[i]]
            y = [track.boxes[seg.idx_start_incl].xyxy[j]]
            plot_line(fig, x, y, col="black")

            x = [track.boxes[seg.idx_end_incl].xyxy[i]]
            y = [track.boxes[seg.idx_end_incl].xyxy[j]]
            plot_line(fig, x, y, col="black")
    
add_lin_segments(fig, segments, track)
fig.show()