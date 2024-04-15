# reuse_distance

To use the tool, use the following steps
(1) cd ScaleTree  
(2) make  
(3) ./analyze  


You have to 
(1) manually specify your input file (including your index sequence) at Line 90@analyze.c 'std::ifstream in("your path to your input file");'
(2) manually specify your output path at Line 107@analyze.c '_PrintResults("your path to your output file");'

Example input:   
 116024  
 123524  
 523441  
 562352    
 ...  
 ...  
 ...  

Example output:  
  Total data is 7202  
  Total access is 2798  
  Distance 0     100  
  Distance 0 to 1        13  
  Distance 2 to 3        39  
  Distance 4 to 7        27  
  Distance 8 to 15       29  
  Distance 16 to 31      26  
  Distance 32 to 63      42  
  Distance 64 to 127     44  
  Distance 128 to 255    59  
  Distance 256 to 511    104  
  Distance 512 to 1023   168  
  Distance 1024 to 2047  274  
  Distance 2048 to 3071  160  
  Distance 3072 to 4095  114  
  Distance 4096 to 5119  78  
  Distance 5120 to 6143  58  
  Distance 6144 to 7167  1463  
  End tree size is 3499  
