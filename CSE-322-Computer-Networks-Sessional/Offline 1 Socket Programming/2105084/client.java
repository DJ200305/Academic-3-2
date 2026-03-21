import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.util.Arrays;
import java.util.Scanner;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class client {
    public static final String services = "Available Services:\n1. User Status\n2.File Upload\n3.See own uploaded files\n4.Show all public files\n5.Download a file\n6.Request a File\n7.Notifications\n8.Fulfilled Requests\n9.Logout";
    public static void uptoserver(ObjectOutputStream out,ObjectInputStream in,String fpath, int fsize, String fileid, int chunksize) throws IOException, ClassNotFoundException {
        FileInputStream fis = new FileInputStream(fpath);
        int rem = fsize;
        byte [] send = new byte[chunksize];
        while(rem > 0){
                    int sbytes = Math.min(chunksize, rem);
                    int read = fis.read(send, 0, sbytes);
                    if(read == -1){
                        break;
                    }
                    byte [] chunks = Arrays.copyOf(send, read);
                    out.writeObject(fileid);
                    out.flush();
                    out.writeObject(chunks);
                    out.flush();
                    rem -= read;
                    String ack = (String) in.readObject();
                    if(!ack.contains("ACKNOWLEDGED")){
                        System.out.println("Acknowledgment not received for chunk");
                        break;
                    }
                    System.out.println(ack);
        }
        fis.close();
    }
    public static void receivechunk(ObjectInputStream in,ObjectOutputStream out, String fname, String fpath) throws IOException, ClassNotFoundException {
        FileOutputStream fos = new FileOutputStream(fpath);
        int chunksize = (Integer) in.readObject();
        int fsz = (Integer) in.readObject();
        int r = 0;
        while(r < fsz){
            byte[] chunk = (byte[]) in.readObject();   
            fos.write(chunk);
            r += chunk.length;
        }
        out.writeObject("Done");
        out.flush();
        System.out.println("File downloaded successfully at " + fpath);
        fos.close();
    }
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        Socket socket = new Socket("localhost", 6666);
        // System.out.println("Connection established");
        // System.out.println("Remote port: " + socket.getPort());
        // System.out.println("Local port: " + socket.getLocalPort());

        // output buffer and input buffer
        ObjectOutputStream out = new ObjectOutputStream(socket.getOutputStream());
        out.flush();    
        ObjectInputStream in = new ObjectInputStream(socket.getInputStream());

        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter your name: ");
            String name = scanner.nextLine();
            out.writeObject(name);
        String serverResponse = (String) in.readObject();
        System.out.println("Server response: " + serverResponse);
        if(serverResponse.equals("Access Denied")){
            System.out.println("Connection closed by server");
            socket.close();
            scanner.close();
            return;
        }
        while(true){
            System.out.println(services);
            System.out.println("Enter your choice:");
        
            String omsg = scanner.nextLine();
            if(omsg.equals("9")){
                out.writeObject(omsg);
                socket.close();
                scanner.close();
                break;
            }
            if(omsg.equals("1")){
                out.writeObject(omsg);
                String msg1 = (String) in.readObject();
                System.out.println(msg1); 
            }
            if(omsg.equals("2")){
                out.writeObject("2");
                System.out.println("Enter Filename:");
                String fname = scanner.nextLine();
                System.out.println("Enter Access Type:");
                String accessType = scanner.nextLine();
                System.out.println("Enter File Path:");
                String fpath = scanner.nextLine();
                File f = new File(fpath);
                int fsize = (int) f.length();

                out.writeObject(fname);
                out.flush();
                out.writeObject(fsize);
                out.flush();
                out.writeObject(accessType);
                out.flush();
                String fileid = (String) in.readObject();
                System.out.println("Allocated File ID: " + fileid);
                String schunksize = (String) in.readObject();
                System.out.println("Allocated Chunks: " + schunksize);
                int chunksize = (Integer) in.readObject();
                uptoserver(out, in, fpath, fsize, fileid, chunksize);
            }
            if(omsg.equals("3")){
                out.writeObject(omsg);
                String msg3 = (String) in.readObject();
                System.out.println(msg3); 
            }
            if(omsg.equals("4")){
                out.writeObject(omsg);
                String msg4 = (String) in.readObject();
                System.out.println(msg4);
            }
            if(omsg.equals("5")){
                out.writeObject(omsg);
                System.out.println("Enter Filename to download:");
                String fname = scanner.nextLine();
                System.out.println("Enter Path to save the file:");
                String fpath = scanner.nextLine();
                out.writeObject(fname);
                out.flush();
                Protocol response = (Protocol) in.readObject();
                if(response.type.equals("absent")){
                    System.out.println(response.content);
                    continue;
                }
                else if(response.type.equals("present")){
                    System.out.println(response.content);
                    Protocol ready = new Protocol("ready", "Client is ready to receive file");
                    out.writeObject(ready);
                    out.flush();
                }
                receivechunk(in,out, fname, fpath);
            }
            if(omsg.equals("6")){
                out.writeObject(omsg);
                System.out.println("Enter a short description of the file:");
                String desc = scanner.nextLine();
                out.writeObject(desc);
                out.flush();
                System.out.println("Enter Recipient name:");
                String recipient = scanner.nextLine();
                out.writeObject(recipient);
                out.flush();
                String resp = (String) in.readObject();
                System.out.println(resp);
            }
            if(omsg.equals("7")){
                out.writeObject(omsg);
                String msg7 = (String) in.readObject();
                System.out.println(msg7); 
            }
            if(omsg.equals("8")){
                out.writeObject(omsg);
                String msg8 = (String) in.readObject();
                System.out.println(msg8);
            }    
        }
    }
}