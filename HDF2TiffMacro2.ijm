// Batch Convert


  dir1 = getDirectory("C:\Users\Aaryan\Downloads\btp\5022ss");
  //format = getFormat();
  dir2 = getDirectory("C:\Users\Aaryan\Downloads\btp\5022ss ");
  list = getFileList(dir1);
  setBatchMode(true);
  for (i=0; i<list.length; i++) {
     showProgress(i+1, list.length);
    cmd="open" + "=[" + dir1+ list[i] + "]" +"1d 2d 3d";
     run("HDF5...", cmd);
     saveAs("Tiff", dir2+list[i]);
     close();
  }
 
 

