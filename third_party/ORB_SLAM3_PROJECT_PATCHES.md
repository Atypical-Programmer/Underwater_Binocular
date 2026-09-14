# Preserved ORB-SLAM3 checkout

The checkout in this directory is retained as project source rather than a gitlink because the SVO2 stereo integration depends on local changes. Its nested Git repository remains intact.

At refactor start it was already dirty at commit `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`, with modified upstream files and untracked `Examples/Stereo/svo2_stereo.cc` and `include/Platform.h`. Those changes were not reset, rewritten, squashed, or deleted. Review them with:

```powershell
git -C third_party/ORB_SLAM3 status
git -C third_party/ORB_SLAM3 diff
```

The parent repository records the relocation and project integration files; the nested repository owns its own upstream history and working-tree state.
