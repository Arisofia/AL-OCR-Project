// Welcome to your new AL extension.
// Remember that object names and IDs should be unique across all extensions.
// AL snippets start with t*, like tpageext - give them a try and happy coding!

namespace DefaultPublisher.ALProject1;

pageextension 50100 CustomerListExt extends "Customer List"
{
    actions
    {
        addlast(processing)
        {
            action(ScanDocument)
            {
                ApplicationArea = All;
                Caption = 'Scan Document';
                Image = Document;
                Promoted = true;
                PromotedCategory = Process;

                trigger OnAction()
                var
                    OCRServiceMgt: Codeunit "OCR Service Mgt.";
                    InStream: InStream;
                    FileName: Text;
                    ResultText: Text;
                begin
                    if UploadIntoStream('Select Image', '', 'All Files (*.*)|*.*', FileName, InStream) then begin
                        ResultText := OCRServiceMgt.UploadAndProcessOCR(InStream);
                        Message('OCR Result: %1', ResultText);
                    end;
                end;
            }
        }
    }
}