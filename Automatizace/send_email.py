import smtplib, ssl, email
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sender_email = "venetiwow@gmail.com"
receiver_email = "daniel.srp@email.cz"
password = "gjklmkqtzepscrfs"

#Create MIMEMultipart object
msg = MIMEMultipart("alternative")
msg["Subject"] = "multipart test"
msg["From"] = sender_email
msg["To"] = receiver_email
filename = "c:/Users/Danie/Desktop/des.pdf"

#HTML Message Part
html = """
<html><head><meta http-equiv="Content-Type" content="text/html; charset=windows-1252"></head> 
 <body style="width:100%;font-family:arial, &#39;helvetica neue&#39;, helvetica, sans-serif;-webkit-text-size-adjust:100%;-ms-text-size-adjust:100%;padding:0;Margin:0"> 
  <div class="es-wrapper-color" style="background-color:#F7F7F7"><!--[if gte mso 9]>
			<v:background xmlns:v="urn:schemas-microsoft-com:vml" fill="t">
				<v:fill type="tile" color="#f7f7f7"></v:fill>
			</v:background>
		<![endif]--> 
   <table class="es-wrapper" width="100%" cellspacing="0" cellpadding="0" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px;padding:0;Margin:0;width:100%;height:100%;background-repeat:repeat;background-position:center top;background-color:#F7F7F7"> 
     <tbody><tr> 
      <td valign="top" style="padding:0;Margin:0"> 
       <table cellpadding="0" cellspacing="0" class="es-header" align="center" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px;table-layout:fixed !important;width:100%;background-color:transparent;background-repeat:repeat;background-position:center top"> 
         <tbody><tr> 
          <td align="center" style="padding:0;Margin:0"> 
           <table class="es-header-body" align="center" cellpadding="0" cellspacing="0" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px;background-color:transparent;width:600px"> 
             <tbody><tr> 
              <td align="left" style="padding:20px;Margin:0;border-radius:10px 10px 0px 0px;background-color:#3da0c3" bgcolor="#3da0c3"> 
               <table cellpadding="0" cellspacing="0" width="100%" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px"> 
                 <tbody><tr> 
                  <td class="es-m-p0r" valign="top" align="center" style="padding:0;Margin:0;width:560px"> 
                   <table cellpadding="0" cellspacing="0" width="100%" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:separate;border-spacing:0px;border-radius:1px" role="presentation"> 
                     <tbody><tr> 
                      <td align="center" style="padding:0;Margin:0;font-size:0px"><a target="_blank" href="https://nfturtle.io/" style="-webkit-text-size-adjust:none;-ms-text-size-adjust:none;mso-line-height-rule:exactly;text-decoration:underline;color:#3D7781;font-size:14px"><img src="https://i.ibb.co/qkFTRLQ/kryptozelvak2.png" alt="Logo" style="display:block;border:0;outline:none;text-decoration:none;-ms-interpolation-mode:bicubic" width="120" title="Logo" height="165"></a></td> 
                     </tr> 
                   </tbody></table></td> 
                 </tr> 
               </tbody></table></td> 
             </tr> 
           </tbody></table></td> 
         </tr> 
       </tbody></table> 
       <table class="es-content" cellspacing="0" cellpadding="0" align="center" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px;table-layout:fixed !important;width:100%"> 
         <tbody><tr> 
          <td align="center" style="padding:0;Margin:0"> 
           <table class="es-content-body" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px;background-color:#ffffff;border-right:1px solid #4c8aa7;border-left:1px solid #4c8aa7;width:600px" cellspacing="0" cellpadding="0" bgcolor="#ffffff" align="center"> 
             <tbody><tr> 
              <td align="left" bgcolor="#ffffff" style="padding:0;Margin:0;padding-left:20px;padding-right:20px;padding-top:30px;background-color:#ffffff"> 
               <table cellpadding="0" cellspacing="0" width="100%" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px"> 
                 <tbody><tr> 
                  <td align="left" style="padding:0;Margin:0;width:558px"> 
                   <table cellpadding="0" cellspacing="0" width="100%" role="presentation" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px"> 
                     <tbody><tr> 
                      <td align="center" class="es-m-txt-c" style="padding:0;Margin:0;padding-bottom:20px;padding-top:30px"><h1 style="Margin:0;line-height:36px;mso-line-height-rule:exactly;font-family:georgia, times, &#39;times new roman&#39;, serif;font-size:30px;font-style:normal;font-weight:normal;color:#023047"><strong>Thank you for turtle photo</strong><br>You are a real hero!</h1></td> 
                     </tr> 
                     <tr> 
                      <td align="center" style="padding:0;Margin:0;padding-right:15px;font-size:0px"><a target="_blank" href="https://nfturtle.io/" style="-webkit-text-size-adjust:none;-ms-text-size-adjust:none;mso-line-height-rule:exactly;text-decoration:underline;color:#3D7781;font-size:14px"><img class="adapt-img" src="file:///C:/Users/Danie/Desktop/images/gettyimages502266246f238799593734bb3a9dc5230389bd2b0.jpg" alt="" style="display:block;border:0;outline:none;text-decoration:none;-ms-interpolation-mode:bicubic" width="475" height="267"></a></td> 
                     </tr> 
                   </tbody></table></td> 
                 </tr> 
                 <tr> 
                  <td class="es-m-p0r" valign="top" align="center" style="padding:0;Margin:0;width:558px"> 
                   <table width="100%" cellspacing="0" cellpadding="0" role="presentation" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px"> 
                     <tbody><tr> 
                      <td align="center" class="es-m-txt-c" style="padding:0;Margin:0;padding-bottom:5px;padding-top:30px"><h1 style="Margin:0;line-height:36px;mso-line-height-rule:exactly;font-family:georgia, times, &#39;times new roman&#39;, serif;font-size:30px;font-style:normal;font-weight:normal;color:#023047">Your sea turtle is unique!</h1></td> 
                     </tr> 
                     <tr> 
                      <td align="center" style="Margin:0;padding-top:10px;padding-bottom:20px;padding-left:40px;padding-right:40px"><p style="Margin:0;-webkit-text-size-adjust:none;-ms-text-size-adjust:none;mso-line-height-rule:exactly;font-family:arial, &#39;helvetica neue&#39;, helvetica, sans-serif;line-height:23px;color:#666666;font-size:15px">Thanks to your contribution, our database is growing and this helps turtle researchers in their work.<br>As you help in the research and protection of sea turtles in this way, we would like to offer you an incentive to buy a discounted NFT. One NFT has just been generated for each new unique turtle. By purchasing the NFT itself, you are helping to build a protective beach for turtles where it is needed.<br>If you are interested in supporting our project more, you&nbsp;can buy the NFT for the normal&nbsp;price on OpenSea NFT market. Click on the button and you will get to the OpenSea NFT market.</p></td> 
                     </tr> 
                     <tr> 
                      <td align="center" class="es-m-txt-c" style="padding:0;Margin:0;padding-top:5px;padding-bottom:5px;font-size:0"> 
                       <table border="0" width="35%" height="100%" cellpadding="0" cellspacing="0" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px;display:inline-table;width:35% !important" role="presentation"> 
                         <tbody><tr> 
                          <td style="padding:0;Margin:0;border-bottom:1px solid #3d85c6;background:none;height:1px;width:100%;margin:0px"></td> 
                         </tr> 
                       </tbody></table></td> 
                     </tr> 
                     <tr> 
                      <td align="center" class="es-m-txt-c" style="padding:0;Margin:0;padding-top:5px;padding-bottom:5px"><p style="Margin:0;-webkit-text-size-adjust:none;-ms-text-size-adjust:none;mso-line-height-rule:exactly;font-family:arial, &#39;helvetica neue&#39;, helvetica, sans-serif;line-height:40px;color:#666666;font-size:20px">Your special price is: .......<br><br></p></td> 
                     </tr> 
                   </tbody></table></td> 
                 </tr> 
               </tbody></table></td> 
             </tr> 
             <tr> 
              <td align="left" style="Margin:0;padding-top:10px;padding-bottom:40px;padding-left:40px;padding-right:40px"> 
               <table cellpadding="0" cellspacing="0" width="100%" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px"> 
                 <tbody><tr> 
                  <td align="center" valign="top" style="padding:0;Margin:0;width:518px"> 
                   <table cellpadding="0" cellspacing="0" width="100%" role="presentation" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px"> 
                     <tbody><tr> 
                      <td align="center" class="es-m-txt-c" style="padding:0;Margin:0"><span class="es-button-border-4 es-button-border" style="border-style:solid;border-color:#2CB543;background:#3d85c6;border-width:0px;display:inline-block;border-radius:30px;width:auto"><a href="https://nfturtle.io/" class="es-button es-button-3" target="_blank" style="mso-style-priority:100 !important;text-decoration:none;-webkit-text-size-adjust:none;-ms-text-size-adjust:none;mso-line-height-rule:exactly;color:#FFFFFF;font-size:18px;border-style:solid;border-color:#3d85c6;border-width:10px 25px 10px 15px;display:inline-block;background:#3d85c6;border-radius:30px;font-family:arial, &#39;helvetica neue&#39;, helvetica, sans-serif;font-weight:normal;font-style:normal;line-height:22px;width:auto;text-align:center"><!--[if !mso]><!-- --><img src="file:///C:/Users/Danie/Desktop/images/41781618489806584.png" alt="icon" width="26" style="display:inline-block;border:0;outline:none;text-decoration:none;-ms-interpolation-mode:bicubic;vertical-align:middle;margin-right:10px" align="absmiddle" height="32"><!--<![endif]-->Go to OpenSea</a></span></td> 
                     </tr> 
                   </tbody></table></td> 
                 </tr> 
               </tbody></table></td> 
             </tr> 
           </tbody></table></td> 
         </tr> 
       </tbody></table> 
       <table cellpadding="0" cellspacing="0" class="es-footer" align="center" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px;table-layout:fixed !important;width:100%;background-color:transparent;background-repeat:repeat;background-position:center top"> 
         <tbody><tr> 
          <td align="center" style="padding:0;Margin:0"> 
           <table class="es-footer-body" align="center" cellpadding="0" cellspacing="0" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px;background-color:transparent;width:600px"> 
             <tbody><tr> 
              <td align="left" style="Margin:0;padding-left:20px;padding-right:20px;padding-top:25px;padding-bottom:25px;border-radius:0px 0px 10px 10px;background-color:#4c8aa7" bgcolor="#4c8aa7"> 
               <table cellpadding="0" cellspacing="0" width="100%" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px"> 
                 <tbody><tr> 
                  <td align="left" style="padding:0;Margin:0;width:560px"> 
                   <table cellpadding="0" cellspacing="0" width="100%" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px"> 
                     <tbody><tr> 
                      <td align="center" style="padding:0;Margin:0;display:none"></td> 
                     </tr> 
                   </tbody></table></td> 
                 </tr> 
               </tbody></table></td> 
             </tr> 
           </tbody></table></td> 
         </tr> 
       </tbody></table> 
       <table cellpadding="0" cellspacing="0" class="es-content" align="center" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px;table-layout:fixed !important;width:100%"> 
         <tbody><tr> 
          <td class="es-info-area" align="center" style="padding:0;Margin:0"> 
           <table class="es-content-body" align="center" cellpadding="0" cellspacing="0" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px;background-color:transparent;width:600px"> 
             <tbody><tr> 
              <td align="left" style="padding:20px;Margin:0"> 
               <table cellpadding="0" cellspacing="0" width="100%" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px"> 
                 <tbody><tr> 
                  <td align="center" valign="top" style="padding:0;Margin:0;width:560px"> 
                   <table cellpadding="0" cellspacing="0" width="100%" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px"> 
                     <tbody><tr> 
                      <td align="center" style="padding:0;Margin:0;display:none"></td> 
                     </tr> 
                   </tbody></table></td> 
                 </tr> 
               </tbody></table></td> 
             </tr> 
           </tbody></table></td> 
         </tr> 
       </tbody></table> 
       <table cellpadding="0" cellspacing="0" class="es-content" align="center" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px;table-layout:fixed !important;width:100%"> 
         <tbody><tr> 
          <td align="center" style="padding:0;Margin:0"> 
           <table class="es-content-body" align="center" cellpadding="0" cellspacing="0" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px;background-color:transparent;width:600px"> 
             <tbody><tr> 
              <td align="left" style="padding:20px;Margin:0"> 
               <table cellpadding="0" cellspacing="0" width="100%" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px"> 
                 <tbody><tr> 
                  <td align="center" valign="top" style="padding:0;Margin:0;width:560px"> 
                   <table cellpadding="0" cellspacing="0" width="100%" style="mso-table-lspace:0pt;mso-table-rspace:0pt;border-collapse:collapse;border-spacing:0px"> 
                     <tbody><tr> 
                      <td align="center" style="padding:0;Margin:0;display:none"></td> 
                     </tr> 
                   </tbody></table></td> 
                 </tr> 
               </tbody></table></td> 
             </tr> 
           </tbody></table></td> 
         </tr> 
       </tbody></table></td> 
     </tr> 
   </tbody></table> 
  </div>  
 
</body></html>
"""

part = MIMEText(html, "html")
msg.attach(part)

# Add Attachment
with open(filename, "rb") as attachment:
    part = MIMEBase("application", "octet-stream")
    part.set_payload(attachment.read())
   
encoders.encode_base64(part)

# Set mail headers
part.add_header(
    "Content-Disposition",
    "attachment", filename= filename
)
msg.attach(part)

# Create secure SMTP connection and send email
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(
        sender_email, receiver_email, msg.as_string()
    )