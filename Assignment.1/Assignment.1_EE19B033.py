                                 #ASSIGNMENT1_SPICE1

from sys import argv, exit                        #sys is a module containing functions(argv())
                                                  #argv is a function is the list commandline arguments to program      

CIRCUIT = '.circuit'                              #assigning .circuit and .end 
END = '.end'

                                 
if len(argv) != 2:                                #to check if there are two arguments(code & circuit)
    print('\nERROR: %s does not have an <inputfile>!!' % argv[0])
    exit()
                                                  #terminate the operation if arguments are not two
try:
        f=open(argv[1])                           #opening the file 
        lines = f.readlines()                     #reading the each line    
        f.close()
                                                  #closing file after reading
        start = -1;                               #(closing the file is mandetory after reading or writing the file to save changes)
        end = -2
        for line in lines:              
            if CIRCUIT == line[:len(CIRCUIT)]:
                start = lines.index(line) 
                                                  #fecthing netlist file
            elif END == line[:len(END)]:  
                end = lines.index(line)           #end of file and avoid any junk
                break
        if start >=end:
            print('Wrong circuit description!')     #checking for discrepancy if any
            exit(0)                                 
          
        reverse_list=[]
        for line in reversed(lines[start+1:end]):
            token= line.split('#')[0].split()
            word=' '.join(reversed(token))
            print(word)
                                                                                  
except IOError:
    print('Invalid file!')                         #terminate the code if any error occured
    exit() 
                                     


