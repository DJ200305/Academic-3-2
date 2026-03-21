import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.Time;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.UUID;
import java.io.Serializable;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

class metadata{
    String fname;
    int fsize;
    int r_bytes;
    Path fPath;
    metadata(String fname, int fsize, Path fPath){
        this.fname = fname;
        this.fsize = fsize;
        this.r_bytes = 0;
        this.fPath = fPath;
    }
}

class clientfiles implements Serializable {
    String fname;
    String accessType;
    String fileID;
    clientfiles(String fname,String fileID, String accessType){
        
        this.fname = fname;
        this.fileID = fileID;
        this.accessType = accessType;
    }
}
class request implements Serializable{
    String requestID;
    String fname;
    String recipient;
    String requester;
    request(String requestID, String fname, String recipient, String requester){
        this.requestID = requestID;
        this.fname = fname;
        this.recipient = recipient;
        this.requester = requester;
    }
}

class Protocol implements Serializable{
    String type;
    String content;
    Protocol(String type, String content){
        this.type = type;
        this.content = content;
    }
}

public class server {
    public static final int MAX_BUFFER_SIZE = 1024*1024;
    public static final int MIN_CHUNK_SIZE = 50;
    public static final int MAX_CHUNK_SIZE = 100;
    public static final AtomicInteger CURR_BUFFER_SIZE = new AtomicInteger(0);
    public static final List<String> clients = Collections.synchronizedList(new ArrayList<>());
    public static final Set<String> actives =  Collections.synchronizedSet(new HashSet<>());
    public static final Map<String,String> status = Collections.synchronizedMap(new HashMap<>());
    public static final Map<String,metadata> metamap = Collections.synchronizedMap(new HashMap<>());
    public static final Map<String,List<clientfiles>> filemap = Collections.synchronizedMap(new HashMap<>());
    public static final Map<String,request> requestmap = Collections.synchronizedMap(new HashMap<>());
    public static final Map<String,ObjectOutputStream> cstream = Collections.synchronizedMap(new HashMap<>());
    public static final Map<String,List<request>> forall = Collections.synchronizedMap(new HashMap<>());
    public static final Map<String,List<String>> fulfilled = Collections.synchronizedMap(new HashMap<>());
    public static final String UPLOAD_DIR = "clientFiles/";
    public static int r_id = 0;
    
    public static String regexmatch(String desc){
        String regex = "\\b[\\w-]+\\.[A-Za-z0-9]{2,5}\\b";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(desc);
        if(matcher.find()){
            return matcher.group(0);
        }
        return null;
    }

    public static void writelog(String name, String action, String fname, String fstatus){
        Date date = new Date();
        Time time = new Time(date.getTime());
        Path path = Paths.get(UPLOAD_DIR+name+"/"+name+"_log.txt");
        String log = String.format(date.toString() + " " + time.toString() + " - " + action + " - " + fname + " - " + fstatus + "\n");
        try{
            Files.write(path, log.getBytes(), java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
        } catch (IOException e) {
            System.out.println("Error writing log for " + name);
        }
    }
    public static String genuuid(){
        return UUID.randomUUID().toString();
    }
    public static synchronized int generid(){
        return r_id++;
    }
    public static void dispatch(request req){
        synchronized(clients){
            String recipient = req.recipient;
            for(String client:clients){
                if(client.equals(recipient) || recipient.equals("ALL")){
                if(client.equals(req.requester)) continue;
                forall.putIfAbsent(client, new ArrayList<>());
                forall.get(client).add(req);
                // ObjectOutputStream out = cstream.get(client);
                // if(out != null){
                //     forall.remove(client);
                //     StringBuilder sb = new StringBuilder();
                //     sb.append("New file request received:\n");
                //     sb.append("Request ID: ").append(req.requestID).append("\n");
                //     sb.append("File Name: ").append(req.fname).append("\n");
                //     sb.append("Requested by: ").append(req.requester).append("\n");
                //     // try {
                //     //     out.writeObject(sb.toString());
                //     //     out.flush();
                //     // } catch (IOException e) {
                //     //     System.out.println("Error sending request to " + client);
                //     // }    
                // }
            }
            }
        }
    }
    public static void uptodir(ObjectInputStream in,ObjectOutputStream out,Path path, String name, String fname,String fuuid, int fsize, String accessType) throws IOException, ClassNotFoundException {
        Path f1path = path.resolve(fname);
        boolean done = false;
        metadata meta = new metadata(fname, fsize, f1path);
        server.metamap.put(fuuid, meta);
        int r_bytes = 0;
        try {
            while(r_bytes < fsize){
                        String incomingid = (String) in.readObject();
                        if(!incomingid.equals(fuuid)){
                            continue;
                        }
                        byte[] chunk_user = (byte[]) in.readObject();
                        Files.write(f1path, chunk_user, java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
                        r_bytes += chunk_user.length;
                        meta.r_bytes = r_bytes;
                        out.writeObject("ACKNOWLEDGED. Received " + r_bytes + " out of " + fsize + " bytes");
                        out.flush();
                    }
                    if(meta.r_bytes == fsize){
                        server.CURR_BUFFER_SIZE.addAndGet(-fsize);
                        writelog(name, "UPLOAD", fname, "SUCCESS");
                        synchronized(server.requestmap){
                            for(String reqid : server.requestmap.keySet()){
                                request req = server.requestmap.get(reqid);
                                System.out.println("requesting");
                                if(req.fname.equals(fname)){
                                    fulfilled.putIfAbsent(req.requester, new ArrayList<>());
                                    fulfilled.get(req.requester).add("File " + fname + " uploaded by " + name + " is now available.");
                                    System.out.println("fulfilled");
                                    server.requestmap.remove(reqid);
                                }
                            }
                        }
                        clientfiles cfile = new clientfiles(fname, fuuid, accessType);
                        synchronized (server.filemap) {
                              server.filemap.putIfAbsent(name, new ArrayList<>());
                              server.filemap.get(name).add(cfile);
                        }
                        System.out.println("File upload completed for " + name + ": " + fname);
                        done = true;
                    }
                    else{
                        writelog(name, "UPLOAD", fname, "FAILED");
                        System.out.println("File size mismatch for " + name + ": " + fname);
                        Files.deleteIfExists(f1path);
                        server.CURR_BUFFER_SIZE.addAndGet(-fsize);
                        server.metamap.remove(fuuid);
                    }
        } catch (Exception e) {
            if(!done){
                writelog(name, "UPLOAD", fname, "FAILED");
                System.out.println("File upload interrupted for " + name + ": " + fname);
                Files.deleteIfExists(f1path);
                server.CURR_BUFFER_SIZE.addAndGet(-fsize);
                server.metamap.remove(fuuid);
                System.out.println("Current Buffer Size: " + server.CURR_BUFFER_SIZE.get());
            } else {
                e.printStackTrace();
            }
        }
    }
    public static void sendchunks(ObjectOutputStream out, ObjectInputStream in, Path fpath, int fsize, int chunksize) throws IOException, ClassNotFoundException {
        byte[] filedata = Files.readAllBytes(fpath);
        int rem = fsize;
        int offset = 0;
        int sbytes = server.MAX_CHUNK_SIZE;
        out.writeObject(sbytes);
        out.flush();
        out.writeObject(fsize);
        out.flush();
        while(rem > 0){
            int rembytes = Math.min(sbytes, rem);
            byte[] chunk = new byte[rembytes];
            System.arraycopy(filedata, offset, chunk, 0, rembytes);
            out.writeObject(chunk);
            out.flush();
            offset += rembytes;
            rem -= rembytes;
            
            
        }
        String resp = (String) in.readObject();
        if(resp.equals("Done")){
            System.out.println("File sent successfully");
        }
        else{
            System.out.println("Error in file transfer acknowledgment");
        }
    }
    public static int randomgen(){
        return ThreadLocalRandom.current().nextInt(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE + 1);
    }
    public static boolean exceedingmax(int size,AtomicInteger existing){
        return size+existing.get() > MAX_BUFFER_SIZE;
    }
    public static void showstatus(ObjectOutputStream out){
        StringBuilder sb = new StringBuilder();
        sb.append("User                    Status\n");
        for(String user : clients){
            sb.append(user).append("                  ").append(status.get(user)).append("\n");
        }
        try {
            out.writeObject(sb.toString());
        } catch (Exception e) {
            System.out.println("Error sending status");
        }
    }
    public static void logout(String name){
        status.put(name, "offline");
        actives.remove(name);
    }
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        ServerSocket welcomeSocket = new ServerSocket(6666);
        clients.add("Sam");
        clients.add("John");
        clients.add("Alice");
        for(String client : clients) {
            status.put(client, "offline");
        }
        
        while(true) {
            System.out.println("Waiting for connection...");
            Socket socket = welcomeSocket.accept();
            System.out.println("Connection established");
            // System.out.println("Remote port: " + socket.getPort());
            // System.out.println("Local port: " + socket.getLocalPort());
            new serverthread(socket);
        }
        
    }
}

class serverthread implements Runnable {
    private Socket clientsocket;
    private String name;
    Thread t;
    public serverthread(Socket socket) {
        this.clientsocket = socket;
        t = new Thread(this);
        t.start();
    }

    @Override
    public void run() {
        
        try {
            ObjectOutputStream out = new ObjectOutputStream(clientsocket.getOutputStream());
            out.flush();
            ObjectInputStream in = new ObjectInputStream(clientsocket.getInputStream());
            String name = (String) in.readObject();
            if(!server.clients.contains(name)){
                out.writeObject("Access Denied");
                out.flush();
                clientsocket.close();
                return;
            }
            if(server.actives.contains(name)){
                out.writeObject("Access Denied");
                out.flush();
                clientsocket.close();
                return;
            }
            server.cstream.put(name, out);
            this.name = name;
            Path path = Paths.get("clientFiles/"+name);
            System.out.println("" + name + " connected");
            out.writeObject("Welcome " + name);
            out.flush();
            server.actives.add(name);
            server.status.put(name, "online");
            if(!Files.exists(path)){
                Files.createDirectories(path);
            }
            while(true){
                String option = (String) in.readObject();
                System.out.println("Chosen service by " + name + ": " + option);
                if(option.equals("1")){
                    server.showstatus(out);
                } else if(option.equals("9")){
                    server.cstream.remove(name);
                    System.out.println(name + " logged out");
                    server.logout(name);
                    clientsocket.close();
                    break;
                }
                else if(option.equals("2")){
                    
                    String fname = (String) in.readObject();
                    
                    int fsize = (Integer) in.readObject();
                    String accessType = (String) in.readObject();
                    System.out.println("File size to be uploaded by " + name + ": " + fsize + " bytes");
                    if(server.exceedingmax((int)fsize, server.CURR_BUFFER_SIZE)){
                        out.writeObject("File size exceeds server buffer limit. Upload denied.");
                        out.flush();
                        System.out.println("Upload denied for " + name + " due to buffer limit");
                        continue;
                    } 
                    String fuuid = server.genuuid();
                    server.CURR_BUFFER_SIZE.addAndGet((int)fsize);
                    int chunks = server.randomgen();
                    String schunks = String.valueOf(chunks);
                    out.writeObject(fuuid);
                    out.flush();
                    out.writeObject(schunks);
                    out.flush();
                    out.writeObject(chunks);
                    out.flush();
                    try {
                        server.uptodir(in, out, path, name, fname, fuuid, fsize,accessType);
                    } catch (Exception e) {
                        System.out.println("Error during file upload for " + name + ": " + fname);
                        e.printStackTrace();
                    }
                }    
                else if(option.equals("3")){
                    List<clientfiles> cfiles = server.filemap.get(name);
                    StringBuilder sb = new StringBuilder();
                    sb.append("File Name                 Status\n");
                    if(cfiles != null){
                        for(clientfiles cf : cfiles){
                            sb.append(cf.fname).append("              ").append(cf.accessType).append("\n");
                        }
                    } else {
                        sb.append("No files uploaded yet.\n");
                    }
                    out.writeObject(sb.toString());
                    out.flush();
                }
                else if(option.equals("4")){
                    StringBuilder sb = new StringBuilder();
                    sb.append("User                    File Name                 Status\n");
                    for(String user : server.filemap.keySet()){
                        List<clientfiles> cfiles = server.filemap.get(user);
                        for(clientfiles cf : cfiles){
                            if(cf.accessType.equals("public")){
                                sb.append(user).append("              ").append(cf.fname).append("              ").append(cf.accessType).append("\n");
                            }
                        }
                    }
                    out.writeObject(sb.toString());
                    out.flush();
                }
                else if(option.equals("5")){
                    String fname = (String) in.readObject();
                    boolean filefound = false;
                    System.out.println("Size: " + server.filemap.keySet().size());
                    for(String user : server.filemap.keySet()){
                        List<clientfiles> cfiles = server.filemap.get(user);
                        for(clientfiles cf : cfiles){
                            if(cf.fname.equals(fname) && cf.accessType.equals("public")){
                                filefound = true;
                                metadata meta = null;
                                for(String key : server.metamap.keySet()){
                                    if(key.equals(cf.fileID)){
                                        meta = server.metamap.get(key);
                                        break;
                                    }
                                }
                                if(meta != null && Files.exists(meta.fPath) && Files.isRegularFile(meta.fPath)){
                                    Protocol message = new Protocol("present", "File found. Preparing to send.");
                                    out.writeObject(message);
                                    out.flush();
                                    Protocol ack = (Protocol) in.readObject();
                                    if(ack.type.equals("ready")){
                                    server.sendchunks(out, in, meta.fPath, meta.fsize, server.MAX_CHUNK_SIZE);
                                    server.writelog(name, "DOWNLOAD", fname, "SUCCESS");
                                    }
                                } 
                                break;
                            }
                        }
                        if(!filefound || server.filemap.keySet().size() == 0){
                            String type = "absent";
                            String msg = "File not in server. You have to make a request";
                            Protocol message = new Protocol(type, msg);
                            out.writeObject(message);
                            out.flush();
                        }
                        else{
                            break;
                        }
                    }
                    System.out.println("in2");
                }
                else if(option.equals("6")){
                    String desc = (String) in.readObject();
                    String fname = server.regexmatch(desc);
                    String requestID = server.genuuid();
                    String recipient = (String) in.readObject();
                    if(fname == null){
                        out.writeObject("No valid filename was found");
                        out.flush();
                        continue;
                    }
                    request req = new request(requestID, fname, recipient, name);
                    server.requestmap.put(requestID, req);
                    server.dispatch(req);
                    out.writeObject("Request dispatched with Request ID: " + requestID + " to " + recipient + ". Please wait.");
                    out.flush();
                }
                else if(option.equals("7")){
                    List<request> pending = server.forall.get(name);
                    if(pending != null && pending.size() > 0){
                        StringBuilder sb = new StringBuilder();
                        sb.append("You have pending file requests:\n");
                        for(request req : pending){
                            sb.append("Request ID: ").append(req.requestID).append("\n");
                            sb.append("File Name: ").append(req.fname).append("\n");
                            sb.append("Requested by: ").append(req.requester).append("\n");
                        }
                        out.writeObject(sb.toString());
                        out.flush();
                        server.forall.remove(name);
                    }  
                    else{
                        out.writeObject("No pending requests.");
                        out.flush();
                    }
                } 
                else if(option.equals("8")){
                    List<String> fmsgs = server.fulfilled.get(name);
                    if(fmsgs != null && fmsgs.size() > 0){
                        StringBuilder sb = new StringBuilder();
                        sb.append("Notifications:\n");
                        for(String msg : fmsgs){
                            sb.append(msg).append("\n");
                        }
                        out.writeObject(sb.toString());
                        out.flush();
                        server.fulfilled.remove(name);
                    }
                    else{
                        out.writeObject("No new requests were made.");
                        out.flush();
                    }
                }
            }
        } catch (Exception e) {
            if(e instanceof IOException){
                System.out.println(this.name + " disconnected");
                server.logout(this.name);
                e.printStackTrace();
            } else {
                e.printStackTrace();
            }
        }
    }
}
