# Java网络编程
~~~
import.net.*;
//1、定义发送端

class UdpSend
{
        public static void main(String[] args) throws Exception
        {
                //1、创建UDP服务，通过DatagramSocket对象
                DatagramSocket ds = new DatagramSocket();   //若指定发送端口使用DatagramSocket ds = new DatagramSocket(8888);否则每次发送端
                                                            //端口都是由系统随机分配的
                
                //2、确定数据，并封装成数据包。DatagramPacket(byte[] buf, int length, InetAddress address,int port)
                byte[] buf = "udp ge men lai le".getBytes();
                DatagramPacket dp = new DatagramPacket(buf,buf.length,InetAddress.getByName("192.168.1.254"),10000);//将数据发送到IP为192.168.1.254的主机上
                                                                                                                    //并指定接收端口为10000
                //3、通过socket服务，将已有的数据包发送出去，通过send方法
                ds.send(dp);
                
                //4、关闭资源
                ds.close();
        }
}  

//2、定义接受端

/*注意：这是两个独立的应用程序，应该写在不同的class文件中，且都应该带主函数
需求：定义一个应用程序，用于接收UDP协议传输的数据并处理的

思路：
1、定义UDPSocket服务。通常会监听一个端口，其实就是给这个接收网络应用程序定义数字标识，方便于明确哪些数据过来该应用程序可以处理。
2、定义一个数据包(刚开始数据包是空的),用于存储接受到的字节数据。因为数据包对象中有更多的功能可以提取字节数据中的不同数据信息
3、通过Socket服务的receive方法将接收到的数据存入已经定义好的数据包中。
4、通过数据包对象的特有功能，将这些不同的数据取出。打印在控制台上。
*/

class UdpRece
{
        public static void main(String[]  args) throws Exception
        {
                //1、创建UDP Socket，建立端点。
                DatagramSocket ds = new DatagramSocket(10000);   //指定该接收端监听10000端口，也可以理解为该接收端有一个10000的数字标识，用于发送端识别
                while(true)
                {       //循环重复信息
                        //2、定义数据包，用来存储接收到的数据。
                        DatagramPacket dp = new DatagramPacket(buf,buf.length);

                        //3、通过服务的receive方法将收到数据存入数据包中。
                        ds.receive(dp);  //阻塞式方法，没有接收到数据就等待

                        //4、通过数据包DatagramPacket提供的方法获取其中的数据
                        String ip = dp.getAddress().getHostAddress();
                        String data = new String(dp.getData(),0,dp.getLength());    //获取buf中有效长度的数据，如果不指定，则返回1024所有的字符数组，很多为空字符
                        int port = dp.getPort();
                        System.out.println(ip+"::"+data+"::"+port);
                }
                //5、关闭资源
                ds.close();
}
~~~

//注意：有时候会出现java.net.BindException接口绑定异常，要使用的接口已经被使用或者该端口虽然已经结束使用但是还没有被释放

## 更改输入流为键盘输入
~~~
class UdpSend2
{
	public static void main(String[] args) throws Exception
	{
		DatagramSocket ds = new DatagramSocket();
		
		BufferedReader bufr = new BufferedReader(new InputStreamReader(System.in));
		String line = null;
		while((line = bufr.readLine())!= null)
		{
			if("886".equals(line)
				break;
			byte[] buf = line.getBytes();
			
			DatagramPacket dp = new DatagramPacket(buf,buf.length,InetAddress.getByName("192.168.1.254"),10001);
			ds.send(dp);
		}
		ds.close();
	}
}
~~~

~~~
class UdpRece2
{
	public static void main(String[] args) throws Exception
	{
		DatagramSocket ds = new DatagramSocket(10001);
		while(true)
		{
			byte[] buf = new byte[1024];
			DatagramPacket dp = new DatagramPacket(buf,buf.length);
			
			ds.receive(dp);
			
			String ip = dp.getAddress().getHostAddress();
			String data = new String(dp.getData(),0,dp.getLength());
			
			System.out.println(ip+"::"+data);
		}
		//如果用于一直接收就不用关闭ds
	}
}
~~~

## 编写一个聊天程序
需求：有收数据的部分和发数据的部分。
这两部分需要同时执行，那就需要用到多线程技术
一个线程控制收，一个线程控制发。

因为收和发的动作是不一致的，所以要定义两个run方法，而且这两个方法要封装到不同的类中。

~~~
class Send implements Runnable
{
	private DatagramSocket ds;
	public Send(DatagramSocket ds)
	{
		this.ds = ds;
	}
	public void run()
	{
		try
		{
			BufferedReader bufr = new BufferedReader(new InputStreamReader(System.in);
			String line = null;
			while((line = bufr.readLine())!= null)
			{
				if("886".equals(line)
					break;
				
				byte[] buf = line.getBytes();
				
				DatagramPacket dp = new DatagramPacket(buf,buf.length,InetAddress.getByName("192.168.1.255"),1000); //192.168.1.1是网络号，192.168.1.255是广播地址
				ds.send(dp);
			}
		}
		catch(Exception e)
		{
			throw new RunTimeException("发送失败！")；
		}
	}
}

class Rece implements Runnable
{
	private DatagramSocket ds;
	public Rece(DatagramSocket ds)
	{
		this.ds = ds;
	}
	public void run()
	{
		try
		{
			while(true)
			{
				byte[] buf = new byte[1024];
				DatagramPacket dp = new DatagramPacket(buf,buf.length);
				ds.receive();
				
				String ip = dp.getAddress().getHostAddress();
				
				String data = new String(dp.getData(),0,dp.getLength());
			
				System.out.println(ip+"::"+data);
				
			}
		}
		catch(Exception e)
		{
			throw new RunTimeException("接收失败！")；
		}
	}
}
//聊天程序主函数
class ChatDemo
{
        public static void main(String[]  args) throws Exception
		{
			DatagramSocket sendSocket = new DatagramSocket();
			DatagramSocket receScoket = new DatagramSocket(10002);
			
			new Thread(new Send(sendSocket).start();
			new Thread(new Rece(receScoket).start();
			
		}
}
~~~


## TCP传输
分为客户端Socket和服务端SeverScoket
建立连接后，通过Socket中的IO流进行数据的传输

### 客户端：
通过查阅Socket对象，发现在该对象建立时，就需要指定被连接的主机。
因为TCP是面向连接的，所以在建立Socket服务时，就要有服务端存在，并连接成功。形成通路后，就在该通道进行数据的传输。

步骤：
1、创建Socket服务，并指定要连接的主机和端口
2、通过Socket获取输出（输入）流
3、输出流写出数据
4、关闭资源

~~~
import java.io.*;
import java.net.*;
class TcpClient
{
	    public static void main(String[]  args) throws Exception
		{
			//创建客户端的Socket服务，指定目的主机和端口
			Socket s = new Socket("192.168.1.254",10003);    //通路一旦建立，就会有一个Socket流，有输入流和输出流
															//不用建立输入输出流，直接使用Socket流的方法调用便可
			//为了发送数据，应该获取Socket流中的输出流
			OutputStream out = s.getOutputStream();
			
			out.write("Tcp ge men lai le".getBytes());
			s.close();
		}
}
~~~

### 服务端：
注意:服务端是通过获取到客户端Socket对象，使用客户端Socket的输入输出流来和客户端进行通信的。

1、建立服务端的Socket服务，通过ServerSocket();并监听一个端口
2、获取连接过来的客户端对象
    通过Serversocketd 的accept方法。没有连接就会等，所以这个方法是阻塞式的。
3、客户端如果发送过来数据，那么服务端要使用对应的客户端对象，并获取到该客户端的读取流来读取发过来的数据。
4、关闭服务端（可选）

~~~
class TcpServer
{
	    public static void main(String[]  args) throws Exception
		{
			//建立服务端的Socket服务，并监听一个端口。
			ServerSocket ss = new ServerSocket(10003);
			
			//通过accept方法获取连接过来的客户端对象
			Socket s = ss.accept();
			
			String ip = s.getInetAddress().getHostAddress();
			System.out.println(ip);
			
			//获取客户端发送过来的数据，那么要使用客户端对象的流来读取数据
			Inputstream in = s.getInputStream();

			
			byte[] buf = new byte[1024];
			int len = in.read(buf);
			
			System.out.println(new String(buf,0,len));
			
			s.close();  //关闭客户端，防止占用服务端资源
			ss.close();
		}
}
~~~
注意：启动的时候要先启动服务端，因为TCP是面向连接的，服务端没有开启的话客户端连不上

## 演示TCP的传输的客户端和服务端的互访
 需求：客户端给服务端发送数据，服务端收到后，给客户端反馈信息。

客户端：
1、建立Socket服务，指定要连接主机和端口
2、获取socket流中的输出流，将数据写到该流中，通过网络发送给服务端
3、获取socket流中的输入流，将服务端反馈的数据获取到，并打印
4、关闭客户端资源
~~~
class TcpClient2
{
	    public static void main(String[]  args) throws Exception
		{
			Socket s = new Socket("192.168.1.254",10004);
			
			OutputStream out = s.getOutputStream();
			out.write("服务端，你好".getBytes());
			
			InputStream in = s.getInputStream();
			
			byte[] buf = new byte[1024];
			int len = in.read(buf);
			
			System.out.println(new String(buf,0,len));
			
			s.close();
		}
}

class TcpServer2
{
		 public static void main(String[]  args) throws Exception
		 {
			 ServerSocket ss = new ServerSocket(10004);
			 Socket s = ss.accept();
			 
			 String ip = s.getInetAddress().getHostAddress();
			 System.out.println(ip);
			 
			 InputStream in = s.getInputStream();
			 byte[] buf = new byte[1024];
			 int len = in.read(buf);
			 System.out.println(new String(buf,0,len));
			 
			 OutputStream out = s.getOutputStream();
			 
			 out.write("哥们收到，你也好".getBytes());
		 }
}
~~~

## 建立一个文本转换服务器
客户端给服务端发送文本，服务端会将文本转成大写返回给客户端
而且客户端可以不断的进行文本转换，当客户端输入over时，转换结束。

分析：
客户端：
源：键盘录入目的：网络设备，网络输出流
操作的是文本数据，选择字符流。

步骤：
1、建立服务端的Socket服务
2、获取键盘录入
3、将数据发送给服务端
4、服务端返回的大写数据
5、关闭资源

都是文本数据，可以使用字符流进行操作，同时提高效率，使用缓冲
~~~
import java.io.*;
import java.net.*;

class TransClient
{
	public static void main(String[]  args) throws Exception
	{
		Socket s = new Socket("192.168.1.254",10005);
		//定义读取键盘数据的流对象
		BufferedReader bufr = new BufferedReader(new InputStreamReader(System.in));
		//定义目的，将数据写入到Socket输出流，发给服务端
		BufferedWriter bufOut = new BufferedWriter(new OutputStreamWriter(s.getOutputStream()));
		//定义一个Socket读取流，读取服务端返回的大写信息。
		BufferedReader bufIn = new BufferedReader(new InputStreamReader(s.getInputStream()));
		
		String line = null;
		while((line = bufr.readLine())!= null)
		{
			if("over".equals(line))
				break;
			bufOut.write(line);
			
			String str = bufIn.readLine();
			System.out.println("Server:"+str);
			
			bufOut.newLine();		//输出一个换行符，因为服务端的readLine()方法需要读到换行符才会有返回值
			
			bufOut.flush();  //将缓冲区中的数据刷新
		}
		s.close();		//客户端断开会返回-1，服务端读到也断开
		ss.close();
	}
}

class TransServer
{
	public static void main(String[]  args) throws Exception
	{
		ServerSocket ss = new ServerSocket(10005);
		Socket s = ss.accept();
		
		String ip = s.getInetAddress().getHostAddress();
		System.out.println(ip);
		
		//读取socket读取流中的数据
		BufferedReader bufIn = new BufferedReader(new InputStreamReader(s.getInputStream()));
		//目的，socket输出流，将大写数据写入到socket输出流，并发送给客户端
		BufferedWriter bufOut = new BufferedWriter(new OutputStreamWriter(s.getOutputStream()));
		
		String line = null;
		while((line = bufrIn.readLine()) != null)
		{
			System.out.println(line);
			bufOut.write(line.toUpperCase());
			
			bufOut.newLine();		//输出一个换行符，因为客户端的readLine()方法需要读到换行符才会有返回值
			
			bufOut.flush();  //将缓冲区中的数据刷新
			
		}
		s.close(); 	 //结束时会返回一个-1
		ss.close();
	}
}
~~~
## TCP复制文件
~~~
class TextClient
{
	public static void main(String[]  args) throws Exception
	{
		Socket s = new Socket("192.168.1.254",10006);
		
		BufferedReader bufr = new BufferedReader(new FileReader("IPDemo.java")));
		
		PrintWriter out = new PrintWriter(s.getOutputStream(),true);
		
		String line = null;
		
		while((line = bufr.readLine())!= null)
		{
			out.println(line);
		}
		s.shutdownOutput(); //关闭客户端的输出流，相当于给流中加入一个结束标记-1，使服务端的读取循环能够结束
		
		BufferedReader bufIn = new BufferedReader(new InputStreamReader(s.getInputStream()));
		String str = bufIn.readLine();
		System.out.println(str);
		
		bufr.close();
		s.close();
		ss.close();
	}
}

class TextServer
{
	public static void main(String[]  args) throws Exception
	{
		ServerSocket ss = new ServerSocket(10006);
		Socket s = ss.accept();
		
		String ip = s.getInetAddress().getHostAddress();
		System.out.println(ip);
		
		BufferedReader bufIn = new BufferedReader(new InputStreamReader(s.getInputStream()));
		
		PrintWriter out = new PrintWriter(new FileWriter("server.txt",true);
		
		String line = null;
		while((line = bufIn.readLine())!= null)
		{
			out.println(line);
		}
		
		PrintWriter pw = new PrintWriter(s.getOutputStream(),true);
		pw.println("上传成功");
		
		out.close();
		s.close();
		ss.close();
	}
}
~~~
## 上传图片
~~~
class PicClient
{
	public static void main(String[]  args) throws Exception
	{
		Socket s = new Socket("192.168.1.254",10007);
		
		FileInputStream fis = new FileInputStream("1.bmp");
		
		OutputStream out = s.getOutputStream();
		byte[] buf = new byte[1024];
		ine len = 0;
		while((len = fis.read(buf)) != -1)
		{
			out.write(buf,0,len);
		}
		
		s.shutdownOutput();		//告诉服务端上传结束，上传一个-1
		
		InputStream in = s.getInputStream();
		
		byte[] bufIn = new byte[1024];
		int num = in.read(bufIn);
		System.out.println(new String(bufIn,0,num));
		
		fis.close();
		s.close();
	}
}

class PicServer
{
	public static void main(String[]  args) throws Exception
	{
		ServerSocket ss = new ServerSocket(10007);
		
		Socket s = ss.accept();
		InputStream in = s.getInputStream();
		FileOutputStream fos = new FileOutputStream("server.bmp");
		
		byte[] buf = new byte[1024];
		
		int len = 0;
		while((len=in.read(buf)) != -1)		//需要获取到客户端上传的停止标记才能停止
		{
			fos.write(buf,0,len);
		}
		
		OutputStream out = s.getOutputStream();
		out.write("上传成功".getBytes());
		
		fos.close();
		s.close();
		ss.close();
	}
}

## 并发上传图片
上面代码的服务端有一个局限性，当A客户端连接上服务端以后，被服务端获取到，服务端执行具体流程
这时B客户连接，服务器端线程正在执行代码，没有回来读到accept()方法，所以在暂时获取不到B客户端对象
那么为了让多个客户端同时并发访问服务端
服务端最好就是将每个客户端封装到一个单独的线程中，这样就可以处理多个客户端请求。
~~~
class PicThread implements Runnable
{
	private Socket s;
	PicThread(Socket s)
	{
		this.s = s;
	}
	public void run()
	{
		try
		{
			int count =1;	//将count定义在run方法内部，如果定为成员变量，则表明是共享变量，多个线程都会修改
			String ip = s.getInetAddress().getHostAddress();
			System.out.println(ip);
			
			InputStream in = s.getInputStream();
			
			File file = new File(ip+"("+(count)+")"+".jpg");
			
			while(file.exists())
				file = new File(ip+"("+(count)+")"+".jpg");
			
			FileOutputStream fos = new FileOutputStream(file);	//封装为文件对象
			
			byte[] buf = new byte[1024];
			
			int len = 0;
			while((len=in.read(buf)) != -1)		//需要获取到客户端上传的停止标记才能停止
			{
				fos.write(buf,0,len);
			}
			
			OutputStream out = s.getOutputStream();
			out.write("上传成功".getBytes());
			
			fos.close();
			s.close();
			ss.close();
		}
		catch(Exception e)
		{
			throw new RuntimeException("上传失败");
		}
	}
}

class PicServer
{
	public static void main(String[]  args) throws Exception
	{
		ServerSocket ss = new ServerSocket(10007);
		
		while(true)
		{
			Socket s = ss.accept();
			
			new Thread(new PicThread(s)).start();
		}
	}
}

class PicClient	//输入参数指定上传文件的路径
{
	public static void main(String[]  args) throws Exception
	{
		if(args.length != 1)
		{
			System.out.println("请选择一个jpg格式的图片");
			return;
		}
		
		File file = new File(args[0]);
		if(!(file.exists() && file.isFile()))
		{
			System.out.println("该文件有问题，要么不存在，要么不是文件");
			return;
		}
		
		if(!file.getName().endsWith(".jpg")
		{
			System.out.println("图片格式错误，请重新选择");
			return;
		}
		
		if(file.length()>1024*1024*5)
		{
			System.out.println("图片太大");
			return;
		}
		Socket s = new Socket("192.168.1.254",10007);
		
		FileInputStream fis = new FileInputStream("1.bmp");
		
		OutputStream out = s.getOutputStream();
		byte[] buf = new byte[1024];
		ine len = 0;
		while((len = fis.read(buf)) != -1)
		{
			out.write(buf,0,len);
		}
		
		s.shutdownOutput();		//告诉服务端上传结束，上传一个-1
		
		InputStream in = s.getInputStream();
		
		byte[] bufIn = new byte[1024];
		int num = in.read(bufIn);
		System.out.println(new String(bufIn,0,num));
		
		fis.close();
		s.close();
	}
}

















