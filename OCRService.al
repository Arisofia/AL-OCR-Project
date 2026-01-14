namespace DefaultPublisher.ALProject1;

using System.RestClient;
using System.IO;

codeunit 50100 "OCR Service Mgt."
{
    var
        APIUrlTxt: Label 'http://localhost:8000/ocr', Locked = true;
        APIKeyTxt: Label 'default_secret_key', Locked = true;

    procedure UploadAndProcessOCR(FileStream: InStream): Text
    var
        Client: HttpClient;
        RequestContent: MultipartFormDataContent;
        ImageContent: HttpContent;
        Response: HttpResponseMessage;
        ResultText: Text;
        Headers: HttpHeaders;
    begin
        ImageContent.WriteFrom(FileStream);
        Headers.Clear();
        ImageContent.GetHeaders(Headers);
        Headers.Remove('Content-Type');
        Headers.Add('Content-Type', 'image/jpeg');

        RequestContent.Add(ImageContent, 'file', 'document.jpg');

        Client.DefaultRequestHeaders().Add('X-API-KEY', APIKeyTxt);

        if Client.Post(APIUrlTxt, RequestContent, Response) then begin
            Response.Content().ReadAs(ResultText);
            exit(ResultText);
        end;

        error('OCR Service call failed: %1', Response.HttpStatusCode());
    end;
}
