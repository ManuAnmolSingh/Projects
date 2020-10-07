package finalProJava;

import java.util.Formatter;
import java.io.FileNotFoundException;


public class WriteFile {
    private static Formatter output;
    
    public static void writerecords(String filename){
        try{
            output=new Formatter(filename);
           }
        catch(FileNotFoundException fnfe){
            System.out.println("error...SecurityException");
            System.exit(1);
        }

         output.format("%10d %10d %10s %10d \n",1,11, "gurupreet",1000 );
         output.format("%10d %10d %10s %10d \n",2,22, "sahar",2000 );
         output.format("%10d %10d %10s %10d\n",3,33, "brahm",3000 );
         output.format("%10d %10d %10s %10d\n",4,44, "manu",4000 );
         output.format("%10d %10d %10s %10d\n",5,55, "anmol",5000 );
         output.format("%10d %10d %10s %10d\n",6,66, "pankaj",6000 );
         output.format("%10d %10d %10s %10d\n",7,77, "sahil",7000 );
         output.format("%10d %10d %10s %10d\n",8,88, "manpreet",8000 );
         output.format("%10d %10d %10s %10d\n",9,99, "kim",9000 );
         output.format("%10d %10d %10s %10d\n",10,100, "tim",10000 );
         output.format("%10d %10d %10s %10d\n",11,111, "vinay",110000 );
         output.format("%10d %10d %10s %10d \n",12,222, "palak",22000 );
         output.format("%10d %10d %10s %10d\n",13,333, "manik",33000 );
         output.format("%10d %10d %10s %10d\n",14,444, "raghav",44000 );
         output.format("%10d %10d %10s %10d\n",15,555, "dilpreet",55000 );
         output.format("%10d %10d %10s %10d\n",16,666, "zahid",66000 );
         output.format("%10d %10d %10s %10d\n",17,777, "jay",77000 );
         output.format("%10d %10d %10s %10d\n",18,888, "sonia",88000 );
         output.format("%10d %10d %10s %10d\n",19,999, "monica",99000 );
         output.format("%10d %10d %10s %10d\n",20,1000, "kritika",100000 );

    }
    public static void closefile(){

           output.close();
    }
    public static void main(String[] args) {
        writerecords("ATM6.txt");
        closefile();
    }

}

