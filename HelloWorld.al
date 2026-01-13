// Welcome to your new AL extension.
// Remember that object names and IDs should be unique across all extensions.
// AL snippets start with t*, like tpageext - give them a try and happy coding!

namespace DefaultPublisher.ALProject1;

pageextension 50100 CustomerListExt extends 22 // "Customer List" (use page ID to avoid localization issues)
{
    trigger OnOpenPage();
    begin
        Message('App published: Hello world');
    end;
}